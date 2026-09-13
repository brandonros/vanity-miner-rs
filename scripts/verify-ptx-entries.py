#!/usr/bin/env python3
"""Require every source kernel entry in an all-features PTX artifact."""
import collections
import hashlib
import json
from pathlib import Path
import re
import sys


def manifest(ptx_path, source_dir):
    expected = []
    for source in sorted(source_dir.glob('*.rs')):
        for definition in source.read_text().split('#[kernel]')[1:]:
            match = re.search(r'\bfn\s+(\w+)\s*\(', definition)
            if not match:
                raise ValueError(f'cannot identify kernel in {source}')
            expected.append(match[1])
    if not expected or len(expected) != len(set(expected)):
        raise ValueError('source kernel inventory is empty or contains duplicates')
    data = ptx_path.read_bytes()
    text = data.decode('utf-8')
    entries = re.findall(r'^\s*(?:\.visible\s+)?\.entry\s+(\w+)\s*\(', text, re.M)
    duplicates = sorted(name for name, count in collections.Counter(entries).items() if count > 1)
    missing = sorted(set(expected) - set(entries))
    extra = sorted(set(entries) - set(expected))
    if duplicates or missing or extra:
        raise ValueError(f'PTX entry mismatch: missing={missing}, extra={extra}, duplicates={duplicates}')
    return {
        'sha256': hashlib.sha256(data).hexdigest(),
        'entry_count': len(entries),
        'self_test_count': sum(name.startswith('kernel_self_test_') for name in entries),
        'entries': sorted(entries),
    }


if __name__ == '__main__':
    try:
        result = manifest(Path(sys.argv[1]), Path(__file__).resolve().parents[1] / 'kernels/src')
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, IndexError) as error:
        sys.exit(f'PTX inventory failed: {error}')
