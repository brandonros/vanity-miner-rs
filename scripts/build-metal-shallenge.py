#!/usr/bin/env python3
"""Build the stock-Rust Shallenge kernel and its LLVM→AIR→Metal artifact bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DEVICE = ROOT / 'crates/kernels/shallenge/metal'
COMMANDS = []

def run(*args, capture=False):
    args = list(map(str, args))
    COMMANDS.append(args)
    return subprocess.run(args, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-metal', type=Path, default=Path(os.environ.get('VANITY_LLVM_METAL_SOURCE', ROOT.parent / 'llvm-metal')))
    parser.add_argument('--output', type=Path, default=ROOT / 'target/metal/shallenge')
    options = parser.parse_args()
    compiler = options.llvm_metal.resolve()
    output = options.output.resolve()
    for name in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'RUSTC_BOOTSTRAP'):
        if os.environ.get(name):
            raise RuntimeError(f'unset {name} for the pinned stock producer')
    rust = run('rustc', '-vV', capture=True)
    if 'release: 1.93.0\n' not in rust or 'LLVM version: 21.1.8\n' not in rust:
        raise RuntimeError('use llvm-metal\'s .#rust-fixtures shell (stable Rust 1.93 / LLVM 21.1.8)')
    if 'LLVM version 21.1.8' not in run('llvm-link', '--version', capture=True):
        raise RuntimeError('LLVM tools must be 21.1.8')
    run('cargo', 'build', '--locked', '--release', '--manifest-path', compiler / 'Cargo.toml', '-p', 'llvm-metal-compiler', '--bin', 'llvm-metalc', '--target-dir', ROOT / 'target/metal/compiler')
    common = ['--locked', '--manifest-path', DEVICE / 'Cargo.toml', '--release', '--target-dir', ROOT / 'target/metal/device']
    run('cargo', 'test', *common)
    start = time.perf_counter()
    messages = run('cargo', 'rustc', *common, '--lib', '--target', 'nvptx64-nvidia-cuda', '--message-format=json', '--', '--emit=llvm-bc', '-Cembed-bitcode=yes', capture=True)
    archives = [Path(f) for line in messages.splitlines() for m in [json.loads(line)]
                if m.get('reason') == 'compiler-artifact' for f in m['filenames'] if f.endswith('.rlib')]
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
        run('opt', '-passes=internalize,globaldce,default<O3>,globaldce,strip-dead-prototypes,verify',
            '-inline-threshold=10000', '-internalize-public-api-list=kernel_shallenge',
            stage / 'linked.bc', '-o', stage / 'kernel.bc')
        undefined = run('llvm-nm', '--undefined-only', stage / 'kernel.bc', capture=True)
        if any(line.split()[-1] not in {'llvm_metal.linear_thread_index', 'llvm_metal.atomic_add_device_u32'} for line in undefined.splitlines()):
            raise RuntimeError(f'unresolved device runtime calls:\n{undefined}')
        run('llvm-dis', stage / 'kernel.bc', '-o', stage / 'kernel.ll')
        frontend_seconds = time.perf_counter() - start
        start = time.perf_counter()
        run(ROOT / 'target/metal/compiler/release/llvm-metalc', 'compile', stage / 'kernel.bc', '--interface', DEVICE / 'kernel.interface.json', '--output', stage)
        lowering_seconds = time.perf_counter() - start
        sources = [DEVICE / 'Cargo.toml', DEVICE / 'Cargo.lock', DEVICE / 'kernel.interface.json', Path(__file__).resolve(), ROOT / 'crates/logic/Cargo.toml', ROOT / 'flake.lock',
                   *sorted((DEVICE / 'src').rglob('*.rs')), *sorted((ROOT / 'crates/logic/src').rglob('*.rs'))]
        names = ['kernel.bc', 'kernel.ll', 'kernel.air.ll', 'kernel.air.bc', 'kernel.bindings.json', 'kernel.metallib']
        report = dict(schema=1, rustc=rust, source_revision=run('git', 'rev-parse', 'HEAD', capture=True).strip(),
                      compiler_source=str(compiler), compiler_flake_lock_sha256=digest(compiler / 'flake.lock'),
                      compiler_sha256=digest(ROOT / 'target/metal/compiler/release/llvm-metalc'),
                      source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                      artifacts={name: digest(stage / name) for name in names}, commands=COMMANDS,
                      timings=dict(frontend_seconds=frontend_seconds, lowering_seconds=lowering_seconds),
                      cache='Cargo may reuse matching artifacts; LLVM linking/lowering rerun', metal_execution='not run by builder')
        (stage / 'kernel.build.json').write_text(json.dumps(report, indent=2)+'\n')
        for name in names + ['kernel.build.json']:
            (stage / name).replace(output / name)
    print(json.dumps(dict(output=str(output), **report['timings'])))

if __name__ == '__main__':
    main()
