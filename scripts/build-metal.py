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
from modes import MODES

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
        'sroa,instcombine<verify-fixpoint;max-iterations=4>,simplifycfg),default<O3>,globaldce,strip-dead-prototypes,verify',
        '-unroll-threshold=1000', '-vectorize-slp=false', '-vectorize-loops=false',
        source, '-o', output, timeout=timeout)
    # Large inlined hash blocks exceed GVN's default backward scan budget.
    # Bounded cleanup exposes stored SHA buffer lengths/domain tags before the
    # unresolved-runtime check. Never replace panic calls or assume their guards.
    cleanup = ",".join(["sroa,early-cse<memssa>,gvn,instcombine<verify-fixpoint;max-iterations=4>,simplifycfg"] * 3)
    run("opt", f"-passes=function({cleanup}),globaldce,strip-dead-prototypes,verify",
        "-memdep-block-scan-limit=10000", output, "-o", output, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-metal', type=Path, default=Path(os.environ.get('VANITY_LLVM_METAL_SOURCE', ROOT.parent / 'llvm-metal')))
    parser.add_argument('--mode', choices=MODES + ['self-test-' + mode for mode in MODES], default='shallenge')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', help='Build one named self-test case into an explicit --output directory')
    parser.add_argument('--monolithic-self-test', action='store_true', help='Build the legacy whole-group self-test kernel')
    parser.add_argument('--inlining', choices=['all', 'retain-scalar', 'selective'], default='selective', help='Helper-retention policy (default: selective); use all to force full inlining')
    options = parser.parse_args()
    is_self_test = options.mode.startswith('self-test-')
    if (options.case or options.monolithic_self_test) and not is_self_test:
        parser.error('--case and --monolithic-self-test require a self-test mode')
    if options.case and (not options.output or options.monolithic_self_test):
        parser.error('--case requires explicit --output and cannot be combined with --monolithic-self-test')
    compiler = options.llvm_metal.resolve()
    # Nix store sources share normalized mtimes. Cargo can otherwise retain old
    # native build-script objects when the compiler pin changes in this checkout.
    compiler_key = hashlib.sha256(str(compiler).encode()).hexdigest()[:16]
    compiler_target = ROOT / 'target/metal/compiler' / compiler_key
    compiler_binary = compiler_target / 'release/llvm-metalc'
    device = ROOT / f'crates/kernels/{options.mode}'
    device_target = ROOT / 'target/metal/device' / options.mode
    output = (options.output or ROOT / 'target/metal' / options.mode).resolve()
    sources = [device / 'Cargo.toml', device / 'Cargo.lock', Path(__file__).resolve(), ROOT / 'crates/logic/Cargo.toml', ROOT / 'flake.lock', ROOT / 'Cargo.toml', ROOT / 'Cargo.lock', ROOT / 'scripts/modes.py',
                   *sorted((device / 'src').rglob('*.rs')),
                   *sorted(p for p in (ROOT / 'vendor/crypto-bigint').rglob('*') if p.is_file()),
                   *sorted(p for p in (ROOT / 'vendor/sec1').rglob('*') if p.is_file()), *sorted((ROOT / 'crates/logic/src').rglob('*.rs'))]
    sources.extend(sorted((device / 'examples').rglob('*.rs')))
    sources.extend(sorted((ROOT / 'crates/kernels/common').glob('candidate_*.rs')))
    if is_self_test:
        sources.append(ROOT / 'crates/kernels/common/metal_self_test_inventory.rs')
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    source_revision = run('git', 'rev-parse', 'HEAD', capture=True).strip()
    for name in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'RUSTC_BOOTSTRAP'):
        if os.environ.get(name):
            raise RuntimeError(f'unset {name} for the pinned stock producer')
    rust = run('rustc', '-vV', capture=True)
    if 'release: 1.98.1\n' not in rust or 'LLVM version: 22.1.8\n' not in rust:
        raise RuntimeError('use the repository\'s default Nix shell (stable Rust 1.98 / LLVM 22.1.8)')
    if 'LLVM version 22.1.8' not in run('llvm-link', '--version', capture=True):
        raise RuntimeError('LLVM tools must be 22.1.8')
    run('cargo', 'build', '--locked', '--release', '--manifest-path', compiler / 'Cargo.toml', '-p', 'llvm-metal-compiler', '--bin', 'llvm-metalc', '--target-dir', compiler_target)
    common = ['--locked', '--manifest-path', device / 'Cargo.toml', '--release', '--target-dir', device_target]
    run('cargo', 'test', *common)
    entry = None
    inventory = []
    if is_self_test:
        listing = run('cargo', 'run', *common, '--example', 'inventory', capture=True)
        for row in listing.splitlines():
            slot, name, group, case_entry = row.split('\t')
            if entry is None:
                entry = group
            if group != entry:
                raise RuntimeError('self-test inventory contains a different group')
            inventory.append(dict(slot=int(slot), name=name, entry=case_entry))
        if not inventory or len({c['name'] for c in inventory}) != len(inventory):
            raise RuntimeError('empty or duplicate self-test inventory')
        expected_entries = {entry, *(c['entry'] for c in inventory)}
        if options.monolithic_self_test:
            inventory = []
        if options.case:
            inventory = [case for case in inventory if case['name'] == options.case]
            if len(inventory) != 1:
                raise RuntimeError(f'unknown self-test case: {options.case}')
            if (output / 'self-tests.json').exists():
                raise RuntimeError('--case output must not overwrite an existing group bundle')
    start = time.perf_counter()
    messages = run('cargo', 'rustc', *common, '--lib', '--target', 'nvptx64-nvidia-cuda', '--config', 'target.nvptx64-nvidia-cuda.rustflags=["-Cno-vectorize-slp", "-Cno-vectorize-loops"]', '--message-format=json', '--', '--emit=llvm-bc', '-Cembed-bitcode=yes', capture=True)
    timings = dict(rustc_seconds=time.perf_counter() - start)
    stage_start = time.perf_counter()
    archives = [Path(f) for line in messages.splitlines() for m in [json.loads(line)]
                if m.get('reason') == 'compiler-artifact' for f in m['filenames'] if f.endswith('.rlib') and Path(f).is_relative_to(device_target / 'nvptx64-nvidia-cuda')]
    if not archives:
        raise RuntimeError('no current device archives')
    output.mkdir(parents=True, exist_ok=True)
    if inventory and not options.case:
        (output / 'self-tests.pending.json').write_text(json.dumps(dict(kernel=entry, source_sha256=source_hashes)) + '\n')
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
        linked_hash = digest(stage / 'linked.bc')
        descriptor_start = time.perf_counter()
        run(compiler_binary, 'extract', stage / 'linked.bc', '--output', stage / 'extracted')
        descriptors = json.loads((stage / 'extracted/descriptors.json').read_text())
        if is_self_test:
            if set(descriptors) != expected_entries:
                raise RuntimeError(f'device descriptors differ from native self-test inventory: missing={sorted(expected_entries - set(descriptors))}, unexpected={sorted(set(descriptors) - expected_entries)}')
        else:
            if len(descriptors) != 1:
                raise RuntimeError('production module must declare exactly one entry')
            entry = next(iter(descriptors))
        linked_input = stage / 'extracted/stripped.bc'
        timings['descriptor_seconds'] = time.perf_counter() - descriptor_start
        shared_frontend_seconds = time.perf_counter() - start
        compiler_hash = digest(compiler_binary)
        group_cases = []
        lowering_total = 0.0
        frontend_total = shared_frontend_seconds
        for case in inventory or [None]:
            unit = stage / ('case-' + str(case['slot']) if case else 'unit')
            unit.mkdir()
            unit_entry = case['entry'] if case else entry
            interface = unit / 'kernel.descriptor.json'
            interface.write_text(json.dumps(descriptors[unit_entry], indent=2) + '\n')
            unit_start = time.perf_counter()
            # Select and discard unrelated exported cases before forced inlining.
            # Rust compilation/linking are shared once across this group's cases.
            if options.inlining == 'all':
                run('opt', '-passes=internalize,globaldce,forceattrs,always-inline,default<O3>,globaldce,strip-dead-prototypes,verify',
                    '-force-remove-attribute=noinline', '-force-attribute=alwaysinline',
                    '-inline-threshold=10000', f'-internalize-public-api-list={unit_entry}',
                    '-vectorize-slp=false', '-vectorize-loops=false',
                    linked_input, '-o', unit / 'inlined.bc')
            else:
                run('opt', '-passes=internalize,globaldce,function(sroa,instcombine,simplifycfg,tailcallelim),globaldce,verify',
                    f'-internalize-public-api-list={unit_entry}', linked_input, '-o', unit / 'selected.bc')
                try:
                    run(compiler_binary, 'prepare', unit / 'selected.bc', '--entry', unit_entry,
                        '--output', unit / 'prepared.bc', '--inlining', options.inlining)
                except subprocess.CalledProcessError:
                    shutil.copyfile(unit / 'selected.bc', output / 'rejected-prepare.bc')
                    run('llvm-dis', unit / 'selected.bc', '-o', output / 'rejected-prepare.ll')
                    raise
                run('opt', '-passes=always-inline,default<O3>,globaldce,strip-dead-prototypes,verify',
                    '-vectorize-slp=false', '-vectorize-loops=false',
                    unit / 'prepared.bc', '-o', unit / 'inlined.bc')
            unit_timings = dict(timings, inline_seconds=time.perf_counter() - unit_start)
            stage_start = time.perf_counter()
            post_inline(unit / 'inlined.bc', unit / 'kernel.bc')
            unit_timings['post_inline_seconds'] = time.perf_counter() - stage_start
            destination = output / 'cases' / case['name'] if case and not options.case else output
            destination.mkdir(parents=True, exist_ok=True)
            undefined = run('llvm-nm', '--undefined-only', unit / 'kernel.bc', capture=True)
            if any(line.split()[-1] not in {'llvm_metal.linear_thread_index', 'llvm_metal.atomic_add_device_u32', 'memcmp', 'bcmp'} for line in undefined.splitlines()):
                shutil.copyfile(unit / 'kernel.bc', destination / 'rejected.bc')
                run('llvm-dis', unit / 'kernel.bc', '-o', destination / 'rejected.ll')
                raise RuntimeError(f'unresolved device runtime calls in {unit_entry}:\n{undefined}')
            run('llvm-dis', unit / 'kernel.bc', '-o', unit / 'kernel.ll')
            frontend_seconds = time.perf_counter() - unit_start
            frontend_total += frontend_seconds
            stage_start = time.perf_counter()
            try:
                run(compiler_binary, 'compile', unit / 'kernel.bc', '--descriptor', interface, '--output', unit,
                    '--inlining', options.inlining)
            except subprocess.CalledProcessError:
                for name in ['kernel.bc', 'kernel.ll']:
                    shutil.copyfile(unit / name, destination / ('rejected.' + name.split('.')[-1]))
                raise
            lowering_seconds = time.perf_counter() - stage_start
            lowering_total += lowering_seconds
            if source_hashes != {str(p.relative_to(ROOT)): digest(p) for p in sources}:
                raise RuntimeError('source changed during build; rerun to produce an attributable bundle')
            if compiler_hash != digest(compiler_binary):
                raise RuntimeError('compiler changed during lowering; rerun the build')
            names = ['kernel.descriptor.json'] + ['kernel.bc', 'kernel.ll', 'kernel.air.ll', 'kernel.air.bc', 'kernel.bindings.json', 'kernel.metallib']
            report = dict(schema=1, rustc=rust, source_revision=source_revision, inlining=options.inlining,
                          compiler_source=str(compiler), compiler_flake_lock_sha256=digest(compiler / 'flake.lock'),
                          compiler_sha256=compiler_hash,
                          linked_bitcode_sha256=linked_hash, source_sha256=source_hashes,
                          artifacts={name: digest(unit / name) for name in names}, commands=list(COMMANDS),
                          timings=dict(frontend_seconds=shared_frontend_seconds + frontend_seconds, lowering_seconds=lowering_seconds, **unit_timings),
                          cache='Cargo may reuse matching artifacts; Rust/link shared once per group; per-entry LLVM optimization/lowering rerun' if case else 'Cargo may reuse matching artifacts; LLVM linking/lowering rerun',
                          metal_execution='not run by builder')
            if case:
                report['self_test'] = dict(case, kernel=entry)
                report['shared_frontend_seconds'] = shared_frontend_seconds
            (unit / 'kernel.build.json').write_text(json.dumps(report, indent=2)+'\n')
            if case:
                group_cases.append(dict(case, directory='cases/' + case['name'], manifest_sha256=digest(unit / 'kernel.build.json')))
            for name in names + ['kernel.build.json']:
                (unit / name).replace(destination / name)
        if inventory and not options.case:
            group = dict(schema=1, kind='self-test-group', kernel=entry, cases=group_cases,
                         source_revision=source_revision, source_sha256=source_hashes,
                         compiler_sha256=compiler_hash, rustc=rust,
                         timings=dict(frontend_seconds=frontend_total, lowering_seconds=lowering_total, **timings),
                         cache='Cargo may reuse matching artifacts; one Rust compilation/link per group; each case optimized and lowered separately',
                         metal_execution='not run by builder')
            (stage / 'self-tests.json').write_text(json.dumps(group, indent=2)+'\n')
            (stage / 'self-tests.json').replace(output / 'self-tests.json')
            (output / 'self-tests.pending.json').unlink(missing_ok=True)
        elif options.monolithic_self_test:
            (output / 'self-tests.json').unlink(missing_ok=True)
            (output / 'self-tests.pending.json').unlink(missing_ok=True)
    print(json.dumps(dict(output=str(output), frontend_seconds=frontend_total, lowering_seconds=lowering_total,
                          cases=len(inventory) if inventory else None)))


if __name__ == '__main__':
    main()
