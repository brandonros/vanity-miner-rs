#!/usr/bin/env python3
"""Check mode metadata and compile/test every CPU/Metal mode independently."""
import subprocess
from modes import MODES, ROOT, check_metadata

check_metadata()
for backend in [[], ['metal']]:
    for mode in MODES:
        features = ','.join(backend + [mode])
        print(f'Checking {features}', flush=True)
        subprocess.run(['cargo', 'test', '--locked', '--release', '-p', 'vanity-miner',
                        '--no-default-features', '--features', features,
                        '--lib', 'args::'], cwd=ROOT, check=True)
print('PASS: eight independent CPU and Metal mode builds and argument tests')
