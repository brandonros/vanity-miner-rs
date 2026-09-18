#!/usr/bin/env python3
"""Build matching stock-Rust device/host artifacts, then run the Metal CLI."""
import os
import argparse
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('--mode', choices=['shallenge', 'ethereum', 'bitcoin', 'solana', 'rsa-modulus', 'p256-public-key', 'p256-signature', 'rsa-pss'], default='shallenge')
options, cli_args = parser.parse_known_args()
features = 'metal' if options.mode == 'shallenge' else f'metal,{options.mode}'
subprocess.run([sys.executable, str(root / 'scripts/build-metal.py'), '--mode', options.mode], cwd=root, check=True)
subprocess.run(['cargo', 'build', '--locked', '--release', '-p', 'vanity-miner', '--no-default-features', '--features', features, '--target-dir', str(root / 'target/metal/host')], cwd=root, check=True)
os.chdir(root)
os.execv(root / 'target/metal/host/release/vanity-miner', ['vanity-miner', *cli_args])
