#!/usr/bin/env bash
# Build matching device and host artifacts, then run the Metal CLI.
# Usage: run-metal.sh [--llvm-metal <checkout>] [--mode <mode|self-test|self-test-mode>] [--inlining <policy>] <CLI arguments>
set -euo pipefail
miner_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$miner_root"
if [[ "${1:-}" == "--llvm-metal" ]]; then
    # Compiler development: build llvm-metalc from a checkout instead of the pin.
    compiler_root=$(cd -- "${2:?--llvm-metal requires a checkout path}" && pwd)
    shift 2
    exec nix develop "path:$miner_root" --override-input llvm-metal "path:$compiler_root" \
        --no-write-lock-file --command bash "$miner_root/scripts/run-metal.sh" "$@"
fi
if ! command -v llvm-metalc >/dev/null; then
    exec nix develop "path:$miner_root" --command bash "$miner_root/scripts/run-metal.sh" "$@"
fi
mode=shallenge inlining=selective
while (($#)); do
    case "$1" in
        --mode) mode=${2:?--mode requires a name}; shift 2 ;;
        --inlining) inlining=${2:?--inlining requires a policy}; shift 2 ;;
        *) break ;;
    esac
done
case "$mode" in
    self-test) modes=$(scripts/modes.sh | sed 's/^/self-test-/'); features=metal,self_test ;;
    self-test-*) modes=$mode; features=metal,${mode//-/_} ;;
    *) modes=$mode; features=metal,$mode ;;
esac
for built in $modes; do
    scripts/build-metal.sh --mode "$built" --inlining "$inlining"
done
cargo build --locked --release -p vanity-miner --no-default-features --features "$features" \
    --target-dir target/metal/host
defaults=()
case "$mode" in
    shallenge|ethereum|bitcoin|solana)
        # Baseline search setting for the hash modes; an explicit value wins.
        if [[ " $* " != *" --batch-size"* ]]; then defaults=(--batch-size 65536); fi ;;
esac
exec target/metal/host/release/vanity-miner ${defaults[@]+"${defaults[@]}"} "$@"
