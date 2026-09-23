#!/usr/bin/env bash
# Copy vanity-miner and its PTX to a GPU host over SSH, then run it:
#
#   scripts/vast-run.sh [--run RUN_ID_OR_URL] -- solana-vanity --prefix aaa
#
# Uploads $BINARY (target/release/vanity-miner) and $PTX/*.ptx
# (target/nvptx64-nvidia-cuda/release), or with --run, the runner and PTX
# artifacts of a GitHub Actions run (needs gh auth login). HOST, PORT and
# REMOTE_USER select the host. BATCH_SIZE, THREADS_PER_BLOCK and STACK_SIZE
# pass through to the miner.
set -euo pipefail

RUN=
if [ "${1:-}" = --run ]; then
    RUN=${2#*/runs/}
    RUN=${RUN%%/*}
    shift 2
fi
[ "${1:-}" = -- ] && shift
[ $# -gt 0 ] || { sed -n '2,10p' "$0"; exit 1; }

REMOTE=${REMOTE_USER:-root}@${HOST:-ssh2.vast.ai}
SSH_PORT=${PORT:-37827}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BINARY=${BINARY:-$ROOT/target/release/vanity-miner}
PTX=${PTX:-$ROOT/target/nvptx64-nvidia-cuda/release}
ARCH=$(ssh -p "$SSH_PORT" "$REMOTE" uname -m)

if [ -n "$RUN" ]; then
    DOWNLOAD=$(mktemp -d)
    trap 'rm -rf "$DOWNLOAD"' EXIT
    gh run download "$RUN" --repo brandonros/vanity-miner-rs --dir "$DOWNLOAD" \
        --name "vanity-miner-$ARCH" --name ptx
    BINARY=$DOWNLOAD/vanity-miner-$ARCH/vanity-miner-$ARCH
    PTX=$DOWNLOAD/ptx
    tar -xzf "$PTX/ptx-bundle.tar.gz" -C "$PTX"
fi

# [.] keeps the pattern from matching the shell that runs pkill.
ssh -p "$SSH_PORT" "$REMOTE" "pkill -f '[.]/vanity-miner'; mkdir -p ptx && rm -f ptx/*.ptx"
scp -q -P "$SSH_PORT" "$BINARY" "$REMOTE:vanity-miner"
scp -q -P "$SSH_PORT" "$PTX"/*.ptx "$REMOTE:ptx/"

RUN_ENV="PTX_PATH=ptx"
for var in BATCH_SIZE THREADS_PER_BLOCK STACK_SIZE; do
    [ -z "${!var:-}" ] || RUN_ENV+=" $var=$(printf %q "${!var}")"
done
# The system loader also runs binaries linked against Nix's loader path.
case $ARCH in
    x86_64) LOADER=/lib64/ld-linux-x86-64.so.2 ;;
    *) LOADER=/lib/ld-linux-$ARCH.so.1 ;;
esac
exec ssh -t -p "$SSH_PORT" "$REMOTE" \
    "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; $RUN_ENV $LOADER ./vanity-miner $(printf '%q ' "$@")"
