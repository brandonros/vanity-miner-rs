#!/usr/bin/env bash
# Compile and test every mode on its own, for the CPU and (on macOS) Metal backends.
set -euo pipefail
miner_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$miner_root"
backends=("")
if [[ $(uname) == Darwin ]]; then backends+=("metal,"); fi
for backend in "${backends[@]}"; do
    for mode in $(scripts/modes.sh); do
        echo "Checking $backend$mode"
        cargo test --locked --release -p vanity-miner --no-default-features \
            --features "$backend$mode" --lib args::
    done
done
