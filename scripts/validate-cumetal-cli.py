#!/usr/bin/env python3
"""Bounded CPU/GPU CLI comparisons using immutable executable and module snapshots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ENTRIES = {
    'shallenge': 'kernel_find_better_shallenge_nonce',
    'solana': 'kernel_find_solana_vanity_private_key',
    'ethereum': 'kernel_find_ethereum_vanity_private_key',
    'bitcoin': 'kernel_find_bitcoin_vanity_private_key',
}
BATCH = re.compile(r'CUMETAL_BATCH batch=(\d+) seed=(\d+) candidates=(\d+) matches=(\d+) verified=true guards=intact')


def fingerprint(path):
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def cases(mode):
    positive = ['brandonros', 'ff' * 32] if mode == 'shallenge' else ['', '']
    result = []
    for threads in (1, 31, 32, 33, 257):
        result.append((f'positive-{threads}', threads, 1, 1, 2, positive, 'positive'))
    result.append(('multi-block', 32, 2, 12345, 2, positive, 'positive'))
    negative = {
        'shallenge': ['brandonros', '00' * 32],
        'solana': ['ZZZZZZZZZZ', ''],
        'ethereum': ['ffffffffffff', ''],
        'bitcoin': ['bc1qqqqqqqqqqq', ''],
    }[mode]
    result.append(('nonmatch', 33, 1, 1, 2, negative, 'negative'))
    if mode == 'shallenge':
        for username in ('a', 'x' * 30):
            result.append((f'username-{len(username)}', 1, 1, 2**64 - 1, 2,
                           [username, 'ff' * 32], 'positive'))
    else:
        pattern = {
            'solana': ['BMz', 'K2sG'],
            'ethereum': ['5395', '279a'],
            # The existing CLI validates suffixes as Bech32 strings; this full
            # known address exercises suffix matching without bypassing validation.
            'bitcoin': ['bc1q', 'bc1qr372mh6wxgm0ftvv6sx584nx3dzzn9phwmffzn'],
        }[mode]
        result.append(('prefix-suffix', 1, 1, 1, 1, pattern, 'positive'))
        result.append(('seed-wrap', 1, 1, 2**64 - 1, 2, positive, 'positive'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--modules', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True,
                        help='New output directory; never overwrites an existing report')
    parser.add_argument('--mode', choices=['all', *ENTRIES], default='all')
    parser.add_argument('--timeout', type=int, default=900,
                        help='Per-case bound including first-time Apple Metal compilation')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    modes = list(ENTRIES) if args.mode == 'all' else [args.mode]
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out / 'snapshot'
    snapshot.mkdir()
    cli, library = snapshot / 'vanity-miner', snapshot / 'libcumetal.dylib'
    shutil.copy2(args.cli, cli)
    shutil.copy2(args.library, library)
    provenance = {'cli': fingerprint(cli), 'runtime': fingerprint(library), 'modules': {}}
    for mode in modes:
        entry = ENTRIES[mode]
        for suffix in ('.metal', '.metal.cumetal-abi'):
            name = entry + suffix
            shutil.copy2(args.modules / name, snapshot / name)
            provenance['modules'][name] = fingerprint(snapshot / name)
    report = {'provenance': provenance, 'modes': modes, 'complete': False, 'cases': []}
    report_path = args.out / 'report.json'

    def save():
        temporary = args.out / 'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(report_path)

    save()
    environment = dict(os.environ, CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0',
                       CUMETAL_TRACE_GPU='1')
    for mode in modes:
        command_name = 'shallenge' if mode == 'shallenge' else mode + '-vanity'
        for name, threads, blocks, seed, batches, pattern, expectation in cases(mode):
            command = [str(cli.resolve()), '--cumetal-library', str(library.resolve()),
                       '--module-dir', str(snapshot.resolve()), '--batches', str(batches),
                       '--threads-per-block', str(threads), '--blocks', str(blocks),
                       '--seed', str(seed), '--verify', command_name, *pattern]
            log_path = args.out / f'{mode}-{name}.log'
            started = time.monotonic()
            with log_path.open('w') as log:
                try:
                    result = subprocess.run(command, env=environment, stdout=log,
                                            stderr=subprocess.STDOUT, timeout=args.timeout)
                    exit_code = result.returncode
                except subprocess.TimeoutExpired:
                    exit_code = 'timeout'
            text = log_path.read_text()
            observed = [tuple(map(int, match)) for match in BATCH.findall(text)]
            errors = []
            if exit_code != 0:
                errors.append(f'CLI exit: {exit_code}')
            if len(observed) != batches:
                errors.append(f'Expected {batches} verified batches, found {len(observed)}')
            for index, (batch, actual_seed, count, matches) in enumerate(observed):
                if (batch, actual_seed, count) != (index, (seed + index) % 2**64, threads * blocks):
                    errors.append(f'Unexpected batch identity: {observed[index]}')
                if expectation == 'negative' and matches != 0:
                    errors.append(f'Negative case produced {matches} matches')
                if expectation == 'positive' and (mode != 'shallenge' or index == 0) and matches != count:
                    errors.append(f'Positive case produced {matches}/{count} matches')
            report['cases'].append({'mode': mode, 'name': name, 'command': command,
                                    'exit_code': exit_code, 'seconds': time.monotonic() - started,
                                    'log': log_path.name, 'verified_batches': len(observed),
                                    'errors': errors, 'passed': not errors})
            save()
            print(f'{mode}/{name}: {"PASS" if not errors else "FAIL"} ({len(observed)} batches)', flush=True)
    report['complete'] = True
    report['passed'] = all(case['passed'] for case in report['cases'])
    save()
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
