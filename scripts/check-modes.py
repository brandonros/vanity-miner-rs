#!/usr/bin/env python3
"""Check mode metadata and compile/test every CPU/Metal mode independently."""
import subprocess
import platform
from modes import MODES, ROOT, check_metadata

check_metadata()
backends = [[], ['metal']] if platform.system() == 'Darwin' else [[]]
for backend in backends:
    for mode in MODES:
        features = ','.join(backend + [mode])
        print(f'Checking {features}', flush=True)
        subprocess.run(['cargo', 'test', '--locked', '--release', '-p', 'vanity-miner',
                        '--no-default-features', '--features', features,
                        '--lib', 'args::'], cwd=ROOT, check=True)
print(f'PASS: {len(MODES) * len(backends)} independent mode builds and argument tests')
