#!/usr/bin/env bash
# Build one kernel crate into its Metal bundle with llvm-metalc.
# Usage: build-metal.sh [--mode <mode|self-test-mode>] [--output <directory>] [llvm-metalc build options]
set -euo pipefail
miner_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mode=shallenge
output=
options=()
while (($#)); do
    case "$1" in
        --mode) mode=${2:?--mode requires a name}; shift 2 ;;
        --output) output=${2:?--output requires a directory}; shift 2 ;;
        *) options+=("$1"); shift ;;
    esac
done
device="$miner_root/crates/kernels/$mode"
if [[ ! -f "$device/Cargo.toml" ]]; then
    echo "unknown mode: $mode" >&2
    exit 2
fi
target="$miner_root/target/metal/device/$mode"
cargo_options=(--locked --release --manifest-path "$device/Cargo.toml" --target-dir "$target")
# A bundle is built only from a kernel whose CPU known answers pass.
cargo test "${cargo_options[@]}"
if [[ "$mode" == self-test-* ]]; then
    # The host registry lists the cases; the compiler checks the device declares the same.
    cargo run "${cargo_options[@]}" --example inventory > "$target/cases.json"
    options+=(--cases "$target/cases.json")
fi
exec llvm-metalc build --crate "$device" --target-dir "$target" \
    --output "${output:-$miner_root/target/metal/$mode}" ${options[@]+"${options[@]}"}
