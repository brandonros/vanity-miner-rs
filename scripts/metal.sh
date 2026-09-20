#!/usr/bin/env bash
# The one script: build Metal bundles, run the GPU tests, or run the CLI.
#   metal.sh build <mode|self-test-mode> [--output <directory>] [llvm-metalc build options]
#   metal.sh test [--skip-build] [test name filter]
#   metal.sh run <mode> <CLI arguments>
# Set LLVM_METAL to an llvm-metal checkout to build the compiler from it.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
if ! command -v llvm-metalc >/dev/null; then
    override=()
    if [[ -n "${LLVM_METAL:-}" ]]; then
        override=(--override-input llvm-metal "path:$(cd -- "$LLVM_METAL" && pwd)" --no-write-lock-file)
    fi
    exec nix develop "path:$root" ${override[@]+"${override[@]}"} --command bash "$0" "$@"
fi

build() {
    local mode=$1 output=$root/target/metal/$1 options=()
    shift
    while (($#)); do
        case "$1" in
            --output) output=$2; shift 2 ;;
            *) options+=("$1"); shift ;;
        esac
    done
    local device=$root/crates/kernels/$mode target=$root/target/metal/device/$mode
    local cargo=(--locked --release --manifest-path "$device/Cargo.toml" --target-dir "$target")
    # A bundle is built only from a kernel whose CPU known answers pass.
    cargo test "${cargo[@]}"
    if [[ "$mode" == self-test-* ]]; then
        # The host registry lists the cases; the compiler checks the device declares the same.
        cargo run "${cargo[@]}" --example inventory > "$target/cases.json"
        options+=(--cases "$target/cases.json")
    fi
    llvm-metalc build --crate "$device" --target-dir "$target" --output "$output" \
        ${options[@]+"${options[@]}"}
}

command=${1:?usage: metal.sh build|test|run}
shift
case "$command" in
    build) build "$@" ;;
    test)
        if [[ "${1:-}" == --skip-build ]]; then
            shift
        else
            for manifest in crates/kernels/*/Cargo.toml; do
                build "$(basename "$(dirname "$manifest")")"
            done
        fi
        # One thread: the GPU tests share the device and Metal's pipeline cache.
        cargo test --locked --release -p vanity-miner --all-features --test metal_gpu -- \
            --ignored --nocapture --test-threads=1 "$@" ;;
    run)
        mode=${1:?run requires a mode}
        shift
        build "$mode"
        feature=$mode
        if [[ "$mode" == self-test-* ]]; then feature=${mode//-/_}; fi
        cargo build --locked --release -p vanity-miner --no-default-features \
            --features "metal,$feature" --target-dir target/metal/host
        exec target/metal/host/release/vanity-miner "$@" ;;
    *) echo "usage: metal.sh build|test|run" >&2; exit 2 ;;
esac
