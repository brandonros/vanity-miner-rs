#!/usr/bin/env bash
# Build the device bundles, then run the ignored GPU tests on an Apple GPU.
# Usage: test-gpu.sh [--skip-build] [--suite all|production|self-tests] [--inlining <policy>]
set -euo pipefail
miner_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$miner_root"
if ! command -v llvm-metalc >/dev/null; then
    exec nix develop "path:$miner_root" --command bash "$miner_root/scripts/test-gpu.sh" "$@"
fi
build=1 suite=all inlining=selective
while (($#)); do
    case "$1" in
        --skip-build) build=0; shift ;;
        --suite) suite=${2:?--suite requires a name}; shift 2 ;;
        --inlining) inlining=${2:?--inlining requires a policy}; shift 2 ;;
        *) echo "Usage: $0 [--skip-build] [--suite all|production|self-tests] [--inlining <policy>]" >&2; exit 2 ;;
    esac
done
# The self-test registry is one test module; everything else is production.
case "$suite" in
    all) prefixes=("" self-test-) filter=() ;;
    production) prefixes=("") filter=(--skip self_test::) ;;
    self-tests) prefixes=(self-test-) filter=(self_test::) ;;
    *) echo "unknown suite: $suite" >&2; exit 2 ;;
esac
export VANITY_METAL_BUNDLES=${VANITY_METAL_BUNDLES:-$miner_root/target/metal}
if ((build)); then
    for prefix in "${prefixes[@]}"; do
        for mode in $(scripts/modes.sh); do
            scripts/build-metal.sh --mode "$prefix$mode" --inlining "$inlining" \
                --output "$VANITY_METAL_BUNDLES/$prefix$mode"
        done
    done
fi
# One process and one thread: GPU tests share Metal's pipeline cache and the device.
cargo=(cargo test --locked --release -p vanity-miner --all-features)
"${cargo[@]}" --test metal_gpu -- --ignored --nocapture --test-threads=1 ${filter[@]+"${filter[@]}"}
if [[ "$suite" != self-tests ]]; then
    "${cargo[@]}" --lib -- --ignored --nocapture --test-threads=1 --exact \
        modes::shallenge::device::metal_tests::real_search_updates_target_advances_counters_and_honors_cancellation
fi
