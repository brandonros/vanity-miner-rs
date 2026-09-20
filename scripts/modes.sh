#!/usr/bin/env bash
# The production modes: every kernel crate that is not a self-test group.
set -euo pipefail
miner_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
for manifest in "$miner_root"/crates/kernels/*/Cargo.toml; do
    mode=$(basename "$(dirname "$manifest")")
    [[ "$mode" == self-test-* ]] || echo "$mode"
done
