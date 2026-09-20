#!/usr/bin/env bash
# Build all eight modes, then require exactly one verified winner from each.
# Use --skip-build to reuse existing device bundles (the host is always built).
set -euo pipefail
miner_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$miner_root"
case "${1:-}" in
    ""|--skip-build) ;;
    *) echo "Usage: $0 [--skip-build]" >&2; exit 2 ;;
esac
if ! command -v llvm-metalc >/dev/null; then
    exec nix develop "path:$miner_root" --command bash "$miner_root/scripts/smoke-metal.sh" "$@"
fi

modes=(shallenge ethereum bitcoin solana rsa-modulus p256-public-key p256-signature rsa-pss)
if [[ "${1:-}" != --skip-build ]]; then
    for mode in "${modes[@]}"; do
        scripts/build-metal.sh --mode "$mode"
    done
fi
cargo build --release --locked -p vanity-miner --no-default-features \
    --features metal,shallenge,ethereum,bitcoin,solana,rsa-modulus,p256-public-key,p256-signature,rsa-pss \
    --target-dir target/metal/host
miner="$miner_root/target/metal/host/release/vanity-miner"

umask 077
smoke_dir=$(mktemp -d)
trap 'rm -rf -- "$smoke_dir"' EXIT
openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 \
    -pkeyopt ec_param_enc:named_curve -out "$smoke_dir/p256.pem"
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$smoke_dir/rsa.pem"
printf 'Metal smoke test\n' > "$smoke_dir/message.bin"

run() {
    local mode=$1 batch_size=1 threads=1 marker="[$1] public_key="
    shift
    # RSA needs many prime-pair candidates even with an empty pattern.
    if [[ "$mode" == rsa-modulus ]]; then batch_size=256; threads=32; fi
    if [[ "$mode" == shallenge ]]; then marker='[shallenge] hash='; fi
    echo "=== $mode ==="
    timeout --kill-after=5 "${VANITY_SMOKE_TIMEOUT:-1800}" "$miner" --exit-on-first-match --verify \
        --batch-size "$batch_size" --threads-per-group "$threads" \
        --metal-artifacts "target/metal/$mode" "$@" | tee "$smoke_dir/$mode.log"
    if [[ $(grep -cF "$marker" "$smoke_dir/$mode.log" || true) != 1 ]]; then
        echo "FAIL: $mode did not print exactly one verified match" >&2
        exit 1
    fi
    echo "PASS: $mode"
}

# Empty patterns accept any valid candidate; Shallenge accepts any smaller hash.
run shallenge shallenge --username smoke-test \
    --target-hash ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
run ethereum ethereum-vanity
run bitcoin bitcoin-vanity
run solana solana-vanity
run p256-public-key p256-public-key-vanity
run p256-signature p256-signature-vanity --search-source ephemeral \
    --key "$smoke_dir/p256.pem" --message "$smoke_dir/message.bin"
run rsa-pss rsa-pss-signature-vanity --key "$smoke_dir/rsa.pem" --message "$smoke_dir/message.bin"
run rsa-modulus rsa-modulus-vanity
echo 'PASS: all 8 Metal modes'
