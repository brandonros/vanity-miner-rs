#!/usr/bin/env python3
"""Build a stock-Rust application kernel and its LLVM→AIR→Metal artifact bundle."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = []

def run(*args, capture=False, timeout=None):
    args = list(map(str, args))
    COMMANDS.append(args)
    return subprocess.run(args, cwd=ROOT, check=True, text=True, timeout=timeout,
                          stdout=subprocess.PIPE if capture else None).stdout

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def post_inline(source, output, *, timeout=None):
    # Match llvm-metal's producer: simplify counters and unroll before O3's
    # expensive ScalarEvolution analysis of partly unrolled Bech32 loops.
    run('opt', '-passes=function(loop-simplify,lcssa,loop(indvars),loop-unroll,'
        'sroa,instcombine,simplifycfg),default<O3>,globaldce,strip-dead-prototypes,verify',
        '-unroll-threshold=1000', '-vectorize-slp=false', '-vectorize-loops=false',
        source, '-o', output, timeout=timeout)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-metal', type=Path, default=Path(os.environ.get('VANITY_LLVM_METAL_SOURCE', ROOT.parent / 'llvm-metal')))
    parser.add_argument('--mode', choices=['shallenge', 'ethereum', 'bitcoin', 'solana'], default='shallenge')
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    compiler = options.llvm_metal.resolve()
    device = ROOT / f'crates/kernels/{options.mode}/metal'
    entry = json.loads((device / 'kernel.interface.json').read_text())['entry']
    device_target = ROOT / 'target/metal/device' / options.mode
    output = (options.output or ROOT / 'target/metal' / options.mode).resolve()
    for name in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'RUSTC_BOOTSTRAP'):
        if os.environ.get(name):
            raise RuntimeError(f'unset {name} for the pinned stock producer')
    rust = run('rustc', '-vV', capture=True)
    if 'release: 1.93.0\n' not in rust or 'LLVM version: 21.1.8\n' not in rust:
        raise RuntimeError('use llvm-metal\'s .#rust-fixtures shell (stable Rust 1.93 / LLVM 21.1.8)')
    if 'LLVM version 21.1.8' not in run('llvm-link', '--version', capture=True):
        raise RuntimeError('LLVM tools must be 21.1.8')
    run('cargo', 'build', '--locked', '--release', '--manifest-path', compiler / 'Cargo.toml', '-p', 'llvm-metal-compiler', '--bin', 'llvm-metalc', '--target-dir', ROOT / 'target/metal/compiler')
    common = ['--locked', '--manifest-path', device / 'Cargo.toml', '--release', '--target-dir', device_target]
    run('cargo', 'test', *common)
    start = time.perf_counter()
    messages = run('cargo', 'rustc', *common, '--lib', '--target', 'nvptx64-nvidia-cuda', '--config', 'target.nvptx64-nvidia-cuda.rustflags=["-Cno-vectorize-slp", "-Cno-vectorize-loops"]', '--message-format=json', '--', '--emit=llvm-bc', '-Cembed-bitcode=yes', capture=True)
    timings = dict(rustc_seconds=time.perf_counter() - start)
    stage_start = time.perf_counter()
    archives = [Path(f) for line in messages.splitlines() for m in [json.loads(line)]
                if m.get('reason') == 'compiler-artifact' for f in m['filenames'] if f.endswith('.rlib') and Path(f).is_relative_to(device_target / 'nvptx64-nvidia-cuda')]
    if not archives:
        raise RuntimeError('no current device archives')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='stage-', dir=output) as directory:
        stage = Path(directory)
        modules = []
        for i, archive in enumerate(archives):
            for j, member in enumerate(run('llvm-ar', 't', archive, capture=True).splitlines()):
                if member.endswith('.o'):
                    data = subprocess.check_output(['llvm-ar', 'p', str(archive), member])
                    if not data.startswith(b'BC\xc0\xde'):
                        raise RuntimeError(f'expected NVPTX LLVM bitcode: {archive}:{member}')
                    path = stage / f'{i}-{j}.bc'
                    path.write_bytes(data)
                    modules.append(path)
        run('llvm-link', *modules, '-o', stage / 'linked.bc')
        timings['link_seconds'] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        run('opt', '-passes=internalize,globaldce,default<O3>,globaldce,strip-dead-prototypes,verify',
            '-inline-threshold=10000', f'-internalize-public-api-list={entry}',
            '-vectorize-slp=false', '-vectorize-loops=false',
            stage / 'linked.bc', '-o', stage / 'inlined.bc')
        timings['inline_seconds'] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        post_inline(stage / 'inlined.bc', stage / 'kernel.bc')
        timings['post_inline_seconds'] = time.perf_counter() - stage_start
        undefined = run('llvm-nm', '--undefined-only', stage / 'kernel.bc', capture=True)
        if any(line.split()[-1] not in {'llvm_metal.linear_thread_index', 'llvm_metal.atomic_add_device_u32'} for line in undefined.splitlines()):
            run('llvm-dis', stage / 'kernel.bc', '-o', output / 'rejected.ll')
            raise RuntimeError(f'unresolved device runtime calls:\n{undefined}')
        run('llvm-dis', stage / 'kernel.bc', '-o', stage / 'kernel.ll')
        frontend_seconds = time.perf_counter() - start
        start = time.perf_counter()
        try:
            run(ROOT / 'target/metal/compiler/release/llvm-metalc', 'compile', stage / 'kernel.bc', '--interface', device / 'kernel.interface.json', '--output', stage)
        except subprocess.CalledProcessError:
            for name in ['kernel.bc', 'kernel.ll']:
                shutil.copyfile(stage / name, output / ('rejected.' + name.split('.')[-1]))
            raise
        lowering_seconds = time.perf_counter() - start
        sources = [device / 'Cargo.toml', device / 'Cargo.lock', device / 'kernel.interface.json', Path(__file__).resolve(), ROOT / 'crates/logic/Cargo.toml', ROOT / 'flake.lock', ROOT / 'Cargo.toml', ROOT / 'Cargo.lock',
                   *sorted((device / 'src').rglob('*.rs')), *sorted((ROOT / 'crates/logic/src').rglob('*.rs'))]
        names = ['kernel.bc', 'kernel.ll', 'kernel.air.ll', 'kernel.air.bc', 'kernel.bindings.json', 'kernel.metallib']
        report = dict(schema=1, rustc=rust, source_revision=run('git', 'rev-parse', 'HEAD', capture=True).strip(),
                      compiler_source=str(compiler), compiler_flake_lock_sha256=digest(compiler / 'flake.lock'),
                      compiler_sha256=digest(ROOT / 'target/metal/compiler/release/llvm-metalc'),
                      source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                      artifacts={name: digest(stage / name) for name in names}, commands=COMMANDS,
                      timings=dict(frontend_seconds=frontend_seconds, lowering_seconds=lowering_seconds, **timings),
                      cache='Cargo may reuse matching artifacts; LLVM linking/lowering rerun', metal_execution='not run by builder')
        (stage / 'kernel.build.json').write_text(json.dumps(report, indent=2)+'\n')
        for name in names + ['kernel.build.json']:
            (stage / name).replace(output / name)
    print(json.dumps(dict(output=str(output), **report['timings'])))

if __name__ == '__main__':
    main()
