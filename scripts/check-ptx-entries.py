#!/usr/bin/env python3
"""Reject full-feature CI PTX missing any of the source-defined kernel entries.

This checks exported entry presence, not GPU numerical correctness.
"""
from pathlib import Path
import re
import sys


def entry_names(ptx):
    ptx = re.sub(r'/\*.*?\*/|//[^\n]*', '', ptx, flags=re.S)
    return set(re.findall(r'\.entry\s+([\w$]+)\s*\(', ptx))


def expected_names(root):
    names = set()
    for module in ('solana_vanity', 'bitcoin_vanity', 'ethereum_vanity', 'shallenge', 'self_test', 'codegen_repros'):
        source = (root / 'kernels/src' / (module + '.rs')).read_text()
        names.update(re.findall(r'pub\s+unsafe\s+extern\s+"C"\s+fn\s+(kernel_\w+)\s*\(', source))
    if not names:
        raise RuntimeError('No source kernel names found')
    return names


def main():
    root = Path(__file__).resolve().parents[1]
    expected = expected_names(root)
    actual = entry_names(Path(sys.argv[1]).read_text())
    missing = expected - actual
    if missing:
        raise SystemExit(f'PTX has {len(actual)} entries; missing {len(missing)} expected kernels:\n'
                         + '\n'.join(sorted(missing)))
    print(f'PTX contains all {len(expected)} expected kernels ({len(actual)} total entries)')


if __name__ == '__main__':
    main()
