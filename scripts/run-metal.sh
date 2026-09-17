#!/usr/bin/env bash
set -euo pipefail
miner_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
compiler_options=()
if [[ "${1:-}" == "--llvm-metal" ]]; then
    compiler_root=$(cd -- "${2:?--llvm-metal requires a checkout path}" && pwd)
    compiler_options=(--override-input llvm-metal "path:$compiler_root" --no-write-lock-file)
    shift 2
fi
cd "$miner_root"
exec nix develop "path:$miner_root#metal" "${compiler_options[@]}" --command python3 "$miner_root/scripts/run-metal.py" "$@"
