#!/usr/bin/env bash
set -euo pipefail
miner_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$miner_root"
if [[ -z "${VANITY_LLVM_METAL_SOURCE:-}" ]]; then
    exec nix develop "path:$miner_root" --command python3 "$miner_root/scripts/test-gpu.py" "$@"
fi
exec python3 "$miner_root/scripts/test-gpu.py" "$@"
