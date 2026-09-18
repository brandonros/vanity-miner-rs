"""Production mode catalog. Kernel contracts remain the authority for ABI layout."""
from pathlib import Path
import json
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODES = ["shallenge", "bitcoin", "ethereum", "solana", "p256-public-key",
         "p256-signature", "rsa-modulus", "rsa-pss"]


def check_metadata():
    features = tomllib.loads((ROOT / 'crates/cli/Cargo.toml').read_text())['features']
    actual = {p.name for p in (ROOT / 'crates/kernels').iterdir()
              if (p / 'Cargo.toml').exists() and not p.name.startswith('self-test-')}
    if actual != set(MODES) or len(set(MODES)) != len(MODES):
        raise RuntimeError('production kernel directories differ from the mode catalog')
    inventory = json.loads((ROOT / 'crates/cli/tests/gpu-coverage.json').read_text())
    workflow = (ROOT / '.github/workflows/metal.yaml').read_text()
    matrix = re.search(r'mode: \[(.*?)\]', workflow).group(1)
    if {m.strip() for m in matrix.split(',')} != set(MODES):
        raise RuntimeError('CI kernel matrix differs from the mode catalog')
    smoke = (ROOT / 'scripts/smoke-metal.sh').read_text()
    for mode in MODES:
        module = mode.replace('-', '_')
        for path in [f'crates/kernels/{mode}/src/contract.rs',
                     f'crates/kernels/{mode}/examples/interface.rs',
                     f'crates/kernels/self-test-{mode}/Cargo.toml',
                     f'crates/cli/src/modes/{module}/cpu.rs',
                     f'crates/cli/src/modes/{module}/metal.rs',
                     f'crates/cli/src/modes/{module}/device.rs']:
            if not (ROOT / path).is_file():
                raise RuntimeError(f'missing mode component: {path}')
        if mode not in features or 'self_test_' + module not in features:
            raise RuntimeError(f'missing mode feature: {mode}')
        if not re.search(r'^run ' + re.escape(mode) + r' ', smoke, re.M):
            raise RuntimeError(f'missing explicit smoke command: {mode}')
        if not any(case['owner'] == f'crates/cli/tests/metal/{module}.rs' for case in inventory['cases']):
            raise RuntimeError(f'missing GPU coverage owner: {mode}')


if __name__ == '__main__':
    check_metadata()
    print('PASS: mode metadata, eight adapters, self-tests, smoke commands and CI matrix')
