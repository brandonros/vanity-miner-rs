#!/usr/bin/env python3
"""Warm up a real Solana CLI search, then measure completed batches for a time window."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time


def run(args, threads, blocks, seconds, name):
    command = [str(args.cli.resolve()), '--cumetal-library', str(args.library.resolve()),
               '--module-dir', str(args.modules.resolve()), '--threads-per-block', str(threads),
               '--blocks', str(blocks), '--seed', '1', 'solana-vanity', 'ZZZZZZZZZZ', '']
    messages = queue.Queue()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, env=dict(os.environ,
                               CUMETAL_TRACE_GPU='1', CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0'))
    def reader():
        for line in process.stdout:
            messages.put(line)
        messages.put(None)
    threading.Thread(target=reader, daemon=True).start()
    start = time.monotonic()
    measured_start = None
    completed = 0
    candidates = 0
    gpu_ns = 0
    pending_ns = None
    timings = []
    try:
        with (args.out / (name + '.log')).open('w') as log:
            while True:
                line = messages.get(timeout=600)
                if line is None:
                    raise RuntimeError(f'CLI stopped before measurement completed: {process.poll()}')
                log.write(line)
                if 'CUMETAL_PROVENANCE event=kernel_launch' in line:
                    if 'launch_success=true' not in line or 'source=generic_ptx' not in line:
                        raise RuntimeError(line.strip())
                    pending_ns = int(re.search(r'duration_ns=(-?\d+)', line)[1])
                    if pending_ns <= 0:
                        raise RuntimeError('GPU timestamp unavailable')
                batch = re.search(r'CUMETAL_BATCH batch=(\d+) seed=(\d+) candidates=(\d+) matches=(\d+) verified=false guards=intact', line)
                if not batch:
                    continue
                index, seed, count, matches = map(int, batch.groups())
                if index != completed or seed != index + 1 or count != threads * blocks or pending_ns is None:
                    raise RuntimeError('Unexpected batch identity or missing GPU timing')
                if matches:
                    raise RuntimeError('Rare-prefix benchmark unexpectedly matched; choose another deterministic fixture')
                completed += 1
                now = time.monotonic()
                if completed <= 2:
                    pending_ns = None
                    if completed == 2:
                        measured_start = now
                        warmup = now - start
                    continue
                candidates += count
                gpu_ns += pending_ns
                timings.append(pending_ns)
                pending_ns = None
                elapsed = now - measured_start
                if elapsed >= seconds:
                    break
        result = dict(threads=threads, blocks=blocks, candidates_per_batch=threads*blocks,
                      requested_seconds=seconds, measured_seconds=elapsed, warmup_seconds=warmup,
                      completed_measured_batches=len(timings), candidates=candidates,
                      cli_candidates_per_second=candidates/elapsed,
                      gpu_seconds=gpu_ns/1e9, gpu_candidates_per_second=candidates/(gpu_ns/1e9),
                      command=command, log=name+'.log', gpu_batch_ns=timings,
                      note='Two warmup batches excluded. Window ends at a completed batch; CLI then terminated. CPU full verification disabled; guards and returned-match checks retained.')
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    print(f'{name}: {threads} x {blocks}: {result["cli_candidates_per_second"]:.0f} CLI keys/s, {result["gpu_candidates_per_second"]:.0f} GPU keys/s ({elapsed:.2f}s)', flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cli', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--modules', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--threads', type=int, default=64)
    p.add_argument('--blocks', type=int, default=32)
    p.add_argument('--tune', action='store_true')
    args=p.parse_args()
    if args.seconds <= 0 or not 1 <= args.threads <= 1024 or not 1 <= args.blocks <= 65535:
        p.error('require positive seconds and valid launch dimensions')
    args.out.mkdir(parents=True, exist_ok=False)
    files=[args.cli,args.library,args.modules/'kernel_find_solana_vanity_private_key.metal',args.modules/'kernel_find_solana_vanity_private_key.metal.cumetal-abi']
    report={'inputs':{str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},'trials':[], 'complete':False}
    def save():
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    if args.tune:
        for threads,blocks in [(32,32),(64,16),(128,8),(256,4)]:
            report['trials'].append(run(args,threads,blocks,5,f'tune-{threads}-{blocks}'))
            save()
        best=max(report['trials'],key=lambda x:x['cli_candidates_per_second'])
        for count in [256,4096,16384]:
            threads=best['threads'];blocks=count//threads
            report['trials'].append(run(args,threads,blocks,5,f'tune-{threads}-{blocks}'))
            save()
        best=max(report['trials'],key=lambda x:x['cli_candidates_per_second'])
        args.threads,args.blocks=best['threads'],best['blocks']
    report['sustained']=run(args,args.threads,args.blocks,args.seconds,'sustained')
    for x in files:
        if hashlib.sha256(x.read_bytes()).hexdigest()!=report['inputs'][str(x.resolve())]:
            raise RuntimeError(f'Input changed during benchmark: {x}')
    report['complete']=True
    save()

if __name__=='__main__':
    main()
