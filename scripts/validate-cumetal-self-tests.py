#!/usr/bin/env python3
"""Record a complete CuMetal CLI self-test run with immutable input snapshots."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def fingerprint(path):
    return {'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def inventory(source):
    # Read the actual kernel slot assignments rather than assuming names or
    # trusting a successful process exit. Reject an unfamiliar source layout.
    entries = []
    for name, body in re.findall(
            r'pub unsafe extern "C" fn (kernel_self_test_\w+)\([^)]*\)\s*\{(.*?)(?=\n\}|\Z)',
            source, re.S):
        slots = re.findall(r'^\s*results\[(\d+)\]\s*=', body, re.M)
        if len(slots) != 1:
            raise ValueError(f'{name}: expected exactly one result assignment')
        entries.append((name, int(slots[0])))
    probe = ('kernel_self_test_stub', 0)
    numerical = [item for item in entries if item != probe]
    if (entries.count(probe) != 1 or len(numerical) != 118
            or sorted(slot for _, slot in numerical) != list(range(118))
            or len({name for name, _ in entries}) != 119):
        raise ValueError('require one launch probe and exactly 118 unique numerical slots')
    return [probe, *sorted(numerical, key=lambda item: item[1])]


def validate_output(log, expected):
    observed = [(name, int(slot)) for name, slot in re.findall(
        r'^NUMERICAL_PASS kernel=(\w+) slot=(\d+); other slots and guards intact$',
        log, re.M)]
    errors = []
    if Counter(observed) != Counter(expected):
        errors.append('Numerical results differ from the complete expected inventory')
    launches = re.findall(
        r'^CUMETAL_PROVENANCE event=kernel_launch kernel="(\w+)" .*launch_success=true(?:\s|$)',
        log, re.M)
    if Counter(launches) != Counter(name for name, _ in expected):
        errors.append('Successful GPU launches differ from the expected inventory')
    return observed, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--modules', type=Path, required=True)
    parser.add_argument('--source', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'kernels/src/self_test.rs')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=7200)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    entries = inventory(args.source.read_text())
    inputs = {'vanity-miner': args.cli, 'libcumetal.dylib': args.library,
              'self_test.rs': args.source}
    for name, _ in entries:
        for suffix in ('.metal', '.metal.cumetal-abi'):
            inputs[name + suffix] = args.modules / (name + suffix)
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        parser.error('Missing inputs; no GPU run started:\n' + '\n'.join(missing))
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out / 'snapshot'
    snapshot.mkdir()
    report = {'complete': False, 'passed': False, 'expected': entries, 'inputs': {}}
    for name, source in inputs.items():
        destination = snapshot / name
        shutil.copy2(source, destination)
        report['inputs'][name] = fingerprint(destination)
    # Ensure inventory and snapshot still refer to the same source if another
    # worker changed the original while inputs were copied.
    if inventory((snapshot / 'self_test.rs').read_text()) != entries:
        raise ValueError('source inventory changed while snapshotting')
    command = [str((snapshot / 'vanity-miner').resolve()), '--cumetal-library',
               str((snapshot / 'libcumetal.dylib').resolve()), '--module-dir',
               str(snapshot.resolve()), 'self-test']
    report['command'] = command
    report_path = args.out / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    started = time.monotonic()
    with (args.out / 'self-test.log').open('w') as log:
        try:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout,
                                    env=dict(os.environ, CUMETAL_TRACE_GPU='1',
                                             CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0')).returncode
        except subprocess.TimeoutExpired:
            status = 'timeout'
    observed, errors = validate_output((args.out / 'self-test.log').read_text(), entries)
    if status != 0:
        errors.append(f'CLI exit: {status}')
    report.update(complete=True, passed=not errors, exit_code=status, errors=errors,
                  observed=observed, seconds=time.monotonic() - started)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{"PASS" if not errors else "FAIL"}: {len(observed)}/{len(entries)} entries; {report_path}')
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
