#!/usr/bin/env python3
"""Reject self-tests containing only pointer setup and constant-result stores.

This is a regression gate for completely folded kernels, not a proof that every
intended operation survived optimization. Calls and more complex PTX require
separate inspection. The deliberately constant launch stub is exempt.
"""
import json
from pathlib import Path
import re
import sys


def audit(text):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    rows = []
    for match in re.finditer(r'\.entry\s+(kernel_self_test_\w+)\s*\(', text):
        start = text.index('{', match.end())
        depth = 1
        end = start + 1
        while depth and end < len(text):
            depth += (text[end] == '{') - (text[end] == '}')
            end += 1
        if depth:
            raise ValueError(f'unclosed entry: {match[1]}')
        body = text[start + 1:end - 1]
        body = re.sub(r'\.reg\s+[^;]+;', '', body)
        statements = [s.strip() for s in body.split(';') if s.strip()]
        constants = {}
        stored = []
        simple = True
        for statement in statements:
            move = re.fullmatch(r'mov\.(?:b|u|s)32\s+(%\w+),\s*(-?\d+|%\w+)', statement)
            store = re.fullmatch(r'st\.global\.(?:b|u|s)32\s+\[[^\]]+\],\s*(-?\d+|%\w+)', statement)
            if move:
                constants[move[1]] = constants.get(move[2]) if move[2].startswith('%') else int(move[2])
            elif store:
                stored.append(constants.get(store[1]) if store[1].startswith('%') else int(store[1]))
            elif not re.fullmatch(r'(?:ld\.param\.u64\s+[^;]+|cvta\.to\.global\.u64\s+[^;]+|ret)', statement):
                simple = False
        rows.append({'entry': match[1], 'constant_result': simple and bool(stored) and all(v is not None for v in stored), 'stored_constants': stored if simple else []})
    if not rows or len({r['entry'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate self-test entry inventory')
    folded = [r['entry'] for r in rows if r['constant_result'] and r['entry'] != 'kernel_self_test_stub']
    return {'self_test_entries': len(rows), 'constant_result_tests': folded, 'entries': rows}


if __name__ == '__main__':
    try:
        result = audit(Path(sys.argv[1]).read_text())
        print(json.dumps(result, indent=2))
        if result['constant_result_tests']:
            sys.exit('Constant-result self-tests: ' + ', '.join(result['constant_result_tests']))
    except (ValueError, OSError, IndexError) as error:
        sys.exit(f'PTX self-test audit failed: {error}')
