#!/usr/bin/env bash
# Run every self-test slot on Metal, requiring a pass with no skips.
# Use --skip-build to reuse device bundles (the host is always built).
set -euo pipefail
miner_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$miner_root"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --skip-build ) ]]; then
    echo "Usage: $0 [--skip-build]" >&2
    exit 2
fi
if [[ -z "${VANITY_LLVM_METAL_SOURCE:-}" ]]; then
    exec nix develop "path:$miner_root" --command bash "$miner_root/scripts/test-metal.sh" "$@"
fi

modes=(shallenge ethereum bitcoin solana rsa-modulus p256-public-key p256-signature rsa-pss)
if [[ "${1:-}" != --skip-build ]]; then
    for mode in "${modes[@]}"; do
        python3 scripts/build-metal.py --mode "self-test-$mode"
    done
fi
cargo build --release --locked -p vanity-miner --no-default-features \
    --features metal,self_test --target-dir target/metal/host
miner="$miner_root/target/metal/host/release/vanity-miner"
test_dir=$(mktemp -d)
trap 'rm -rf -- "$test_dir"' EXIT
log=target/metal/self-test.log

# Compare every reported pass with the complete host registry, in slot order.
# The Metal runner also checks each bundle's slot identity and buffer guards.
"$miner" self-test --list | cut -f1 > "$test_dir/expected"
"$miner" --metal-artifacts target/metal self-test 2>&1 | tee "$log"
sed -n 's/^\[Metal\] PASS //p' "$log" > "$test_dir/actual"
diff -u "$test_dir/expected" "$test_dir/actual"
count=$(wc -l < "$test_dir/expected" | tr -d ' ')
grep -Fx "[Metal] self-test: $count passed, 0 failed, 0 skipped" "$log"
for mode in "${modes[@]}"; do
    group=${mode//-/_}
    slots=$(grep -c "^$group\." "$test_dir/actual")
    echo "PASS: $mode ($slots slots)"
done
echo "PASS: all 8 Metal self-test groups ($count slots, no skips)"
