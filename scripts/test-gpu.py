#!/usr/bin/env python3
"""Build and run the coverage inventory on an Apple GPU; fail on missing/unlisted tests."""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
from modes import MODES


def run(command, timeout):
    # Own the process group so a timeout also stops CLI children of Cargo tests.
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def check_discovery(target, output, cases, exclusions):
    found = {line.removesuffix(': test') for line in output.splitlines() if line.endswith(': test')}
    expected = [case['name'] for case in cases if case['target'] == target]
    excluded = [case['name'] for case in exclusions if case['target'] == target]
    declared = expected + excluded
    if len(set(declared)) != len(declared):
        raise RuntimeError(f'{target}: duplicate or excluded GPU case in inventory')
    if found != set(declared):
        raise RuntimeError(f'{target} discovery differs from inventory: missing={sorted(set(declared) - found)}, unlisted={sorted(found - set(declared))}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-build', action='store_true', help='Reuse existing device bundles; compile current host tests and check source hashes')
    parser.add_argument('--list', action='store_true', help='Check Cargo discovery against the inventory without loading Metal')
    parser.add_argument('--suite', choices=['all', 'production', 'self-tests', 'cli'], default='all')
    parser.add_argument('--timeout', type=int, default=10800, help='Deadline in seconds for each build/test command')
    parser.add_argument('--inlining', choices=['selective', 'all', 'retain-scalar'], default='selective', help='Required build policy (default: selective); use all to opt out')
    options = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('GPU execution/discovery requires macOS; CPU references run with --no-default-features')
    if options.timeout <= 0:
        parser.error('--timeout must be positive')
    overrides = [key for key in os.environ if key.startswith('VANITY_METAL_') and key.endswith('ARTIFACTS')]
    if overrides:
        parser.error(f'unset per-mode overrides {overrides}; use VANITY_METAL_BUNDLES for checked bundles')
    bundles = Path(os.environ.get('VANITY_METAL_BUNDLES', ROOT / 'target/metal')).resolve()
    os.environ['VANITY_METAL_BUNDLES'] = str(bundles)
    inventory = json.loads((ROOT / 'crates/cli/tests/gpu-coverage.json').read_text())
    cases = inventory['cases']
    cargo = ['cargo', 'test', '--locked', '--release', '-p', 'vanity-miner', '--all-features']
    targets = {'metal_gpu': ['--test', 'metal_gpu'], 'lib': ['--lib']}
    # Discover every Cargo test executable, including lib-linked/private tests and
    # future integration targets. Running --list never executes test bodies.
    messages = subprocess.check_output(cargo + ['--no-run', '--message-format=json'], cwd=ROOT, text=True, timeout=options.timeout)
    executables = {}
    for line in messages.splitlines():
        message = json.loads(line)
        if message.get('reason') == 'compiler-artifact' and message.get('profile', {}).get('test') and message.get('executable'):
            target = message['target']
            key = 'lib' if 'lib' in target['kind'] else target['name']
            executables[key] = message['executable']
    declared = {case['target'] for case in cases + inventory['excluded_ignored']}
    if not declared <= executables.keys():
        raise RuntimeError(f'missing Cargo test targets: {sorted(declared - executables.keys())}')
    for target, executable in executables.items():
        output = subprocess.check_output([executable, '--ignored', '--list', '--format', 'terse'], cwd=ROOT, text=True, timeout=options.timeout)
        check_discovery(target, output, cases, inventory['excluded_ignored'])
    selected = [case for case in cases if options.suite == 'all' or case['suite'] == options.suite or (options.suite == 'production' and case['suite'] == 'cli')]
    for case in selected:
        print(f"{case['target']}::{case['name']} [{case['owner']}]", flush=True)
    if options.list:
        return
    groups = (MODES if options.suite != 'self-tests' else []) + (['self-test-' + mode for mode in MODES] if options.suite in ['all', 'self-tests'] else [])
    if not options.skip_build:
        for mode in groups:
            run(['python3', 'scripts/build-metal.py', '--mode', mode, '--inlining', options.inlining, '--output', str(bundles / mode)], options.timeout)
    # The loader validates artifact hashes/ABI; verify source provenance here too.
    # CI downloads bundles built in a different checkout, so compare relative source hashes.
    import hashlib
    for mode in groups:
        directory = bundles / mode
        manifests = sorted(directory.rglob('kernel.build.json'))
        if not manifests:
            raise RuntimeError(f'no build manifests in {directory}')
        for path in manifests:
            manifest = json.loads(path.read_text())
            if manifest.get('inlining') != options.inlining:
                raise RuntimeError(f'wrong inlining policy in {path}: expected {options.inlining}')
            for source, digest in manifest['source_sha256'].items():
                if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f'stale bundle {path}: source changed: {source}')
    start = time.monotonic()
    # Run all integration tests in one process to reuse Metal's pipeline cache.
    # Exact name selection for the lib-linked session avoids stochastic CPU tests.
    for target, flags in targets.items():
        names = [case['name'] for case in selected if case['target'] == target]
        if names:
            run(cargo + flags + ['--', '--ignored', '--exact', *names, '--nocapture', '--test-threads=1'], options.timeout)
    print(f'PASS: {len(selected)} inventoried GPU test functions ({time.monotonic() - start:.1f}s); see registry output for named check counts', flush=True)


if __name__ == '__main__':
    main()
