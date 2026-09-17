#!/usr/bin/env python3
"""Build matching stock-Rust device/host artifacts, then run the Metal CLI."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(root / 'scripts/build-metal-shallenge.py')], cwd=root, check=True)
subprocess.run(['cargo', 'build', '--locked', '--release', '-p', 'vanity-miner', '--no-default-features', '--features', 'metal', '--target-dir', str(root / 'target/metal/host')], cwd=root, check=True)
os.chdir(root)
os.execv(root / 'target/metal/host/release/vanity-miner', ['vanity-miner', *sys.argv[1:]])
