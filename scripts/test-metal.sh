#!/usr/bin/env bash
# Compatibility entry for the complete named self-test registry and selection checks.
set -euo pipefail
miner_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "$miner_root/scripts/test-gpu.sh" --suite self-tests "$@"
