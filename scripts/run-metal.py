#!/usr/bin/env python3
"""Build matching stock-Rust device/host artifacts, then run the Metal CLI."""
import os
import argparse
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
from modes import MODES
parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('--mode', choices=MODES + ['self-test'] + ['self-test-' + mode for mode in MODES], default='shallenge')
parser.add_argument('--inlining', choices=['selective', 'all', 'retain-scalar'], default='selective', help='Use all to opt out of selective inlining')
options, cli_args = parser.parse_known_args()
if options.mode in ('shallenge', 'ethereum', 'bitcoin', 'solana'):
    # Baseline search settings; explicit CLI values remain authoritative.
    defaults = []
    for flag, value in [('--batch-size', '65536'), ('--threads-per-group', '64')]:
        if not any(arg == flag or arg.startswith(flag + '=') for arg in cli_args):
            defaults.extend([flag, value])
    cli_args = defaults + cli_args
if options.mode == 'self-test':
    modes = ['self-test-' + mode for mode in MODES]
    features = 'metal,self_test'
elif options.mode.startswith('self-test-'):
    modes = [options.mode]
    features = 'metal,' + options.mode.replace('-', '_')
else:
    modes = [options.mode]
    features = f'metal,{options.mode}'
for mode in modes:
    subprocess.run(['bash', str(root / 'scripts/build-metal.sh'), '--mode', mode, '--inlining', options.inlining], cwd=root, check=True)
subprocess.run(['cargo', 'build', '--locked', '--release', '-p', 'vanity-miner', '--no-default-features', '--features', features, '--target-dir', str(root / 'target/metal/host')], cwd=root, check=True)
os.chdir(root)
os.execv(root / 'target/metal/host/release/vanity-miner', ['vanity-miner', *cli_args])
