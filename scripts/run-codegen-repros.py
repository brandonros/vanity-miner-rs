#!/usr/bin/env python3
"""Run small Rust-generated reproductions through CUDA or CuMetal; report raw mismatches."""
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ptx', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--cumetalc', type=Path, help='Omit for NVIDIA CUDA execution of PTX')
    parser.add_argument('--entry', choices=['kernel_repro_nonce_sequence', 'kernel_repro_alphabet_helper'])
    parser.add_argument('--out', type=Path, required=True, help='New evidence directory')
    args = parser.parse_args()
    rows = json.loads(args.vectors.read_text())
    if args.entry:
        rows = [r for r in rows if r['entry'] == args.entry]
    if not rows:
        parser.error('No selected cases')
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'passed': False, 'cases': [], 'inputs': {}}
    for name, source in [('input.ptx', args.ptx), ('vectors.json', args.vectors),
                         ('runtime' + args.library.suffix, args.library)]:
        target = args.out / name
        shutil.copy2(source, target)
        report['inputs'][name] = digest(target)
    ptx = (args.out / 'input.ptx').resolve()
    library = (args.out / ('runtime' + args.library.suffix)).resolve()
    compiler = None
    if args.cumetalc:
        compiler = (args.out / 'cumetalc').resolve()
        shutil.copy2(args.cumetalc, compiler)
        report['inputs']['cumetalc'] = digest(compiler)
    path = args.out / 'report.json'
    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')
    save()
    os.environ['CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS'] = '0'
    os.environ['CUMETAL_TRACE_GPU'] = '1'
    lib = c.CDLL(str(library))
    ptr, u32, u64 = c.c_void_p, c.c_uint32, c.c_uint64
    def api(name, types, *values):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, c.c_int
        status = fn(*values)
        if status:
            raise RuntimeError(f'{name}: driver error {status}')
    context = ptr()
    try:
        api('cuInit', [u32], 0)
        api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
        for entry in dict.fromkeys(row['entry'] for row in rows):
            module_path = ptx
            if compiler:
                module_path = (args.out / (entry + '.metal')).resolve()
                command = [str(compiler), str(ptx), '--backend=cumetal-ir', '--ptx-strict',
                           '--entry', entry, '--emit=msl', '-o', str(module_path)]
                with (args.out / (entry + '.compile.log')).open('w') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                sidecar = Path(str(module_path) + '.cumetal-abi').read_text().splitlines()
                if [line for line in sidecar if line.startswith('arg ')] != ['arg buffer 8'] * 2:
                    raise RuntimeError(f'{entry}: expected two buffer arguments: {sidecar}')
                report.setdefault('modules', {})[entry] = digest(module_path)
            module, function = ptr(), ptr()
            try:
                api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(module_path))
                api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, entry.encode())
                for index, row in enumerate(rows):
                    if row['entry'] != entry:
                        continue
                    guard = 0xa5a5a5a5a5a5a5a5
                    initial = [guard] * 2 + row['input'] + [guard] * 2
                    expected = [guard] * 2 + row['expected'] + [guard] * 2
                    arrays = [(u64 * len(initial))(*initial),
                              (u64 * len(expected))(*([guard] * len(expected)))]
                    allocations = []
                    try:
                        for data in arrays:
                            address = u64()
                            api('cuMemAlloc', [c.POINTER(u64), c.c_size_t], c.byref(address), c.sizeof(data))
                            allocations.append(address)
                            api('cuMemcpyHtoD', [u64, ptr, c.c_size_t], address, c.cast(data, ptr), c.sizeof(data))
                        pointers = [u64(a.value + 16) for a in allocations]
                        values = (ptr * 2)(*[c.cast(c.pointer(a), ptr) for a in pointers])
                        api('cuLaunchKernel', [ptr] + [u32] * 7 + [ptr, c.POINTER(ptr), ptr],
                            function, 1, 1, 1, 1, 1, 1, 0, None, values, None)
                        api('cuCtxSynchronize', [])
                        for data, address in zip(arrays, allocations):
                            api('cuMemcpyDtoH', [ptr, u64, c.c_size_t], c.cast(data, ptr), address, c.sizeof(data))
                        passed = list(arrays[0]) == initial and list(arrays[1]) == expected
                        result = dict(entry=entry, case=index, input=row['input'], passed=passed)
                        if not passed:
                            result.update(expected=expected, actual=list(arrays[1]), input_readback=list(arrays[0]))
                        report['cases'].append(result)
                        save()
                        if not passed:
                            raise RuntimeError(f'{entry}, case {index}, input {row["input"]}: {result}')
                    finally:
                        for address in allocations:
                            api('cuMemFree', [u64], address)
            finally:
                if module:
                    api('cuModuleUnload', [ptr], module)
        report['passed'] = len(report['cases']) == len(rows) and all(r['passed'] for r in report['cases'])
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        save()
        if context:
            api('cuCtxDestroy', [ptr], context)
    print(f'PASS: {len(rows)} cases; {path}')


if __name__ == '__main__':
    main()
