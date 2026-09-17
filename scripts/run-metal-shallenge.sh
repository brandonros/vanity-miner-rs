#!/usr/bin/env bash
set -euo pipefail
miner_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$miner_root"
exec nix develop "path:$miner_root#metal" --command python3 "$miner_root/scripts/run-metal-shallenge.py" "$@"
