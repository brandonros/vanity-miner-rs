#!/usr/bin/env bash

set -e

PORT=15633
HOST=ssh9.vast.ai
USER=root

LOCAL_BIN="$(dirname "$0")/../target/release/vanity-miner"

if [ ! -x "$LOCAL_BIN" ]; then
    echo "missing local binary at $LOCAL_BIN — run: cargo build --release --features gpu"
    exit 1
fi

echo "=================================================================="
echo "==  SCP :: pushing $(ls -lh "$LOCAL_BIN" | awk '{print $5}') binary to $HOST"
echo "=================================================================="
scp -o StrictHostKeyChecking=no -P "$PORT" "$LOCAL_BIN" "$USER@$HOST:vanity-miner"

ssh -o StrictHostKeyChecking=no -p "$PORT" "$USER@$HOST" <<'EOF'
banner() {
    echo ""
    echo "=================================================================="
    echo "==  $1"
    echo "=================================================================="
}

banner "ENV CHECK :: killall"
if ! command -v killall &> /dev/null; then
    apt update && apt install -y psmisc
else
    echo "killall already installed"
fi

banner "CLEANUP :: stop any running miner"
killall vanity-miner || true
chmod +x vanity-miner
ls -lh vanity-miner

banner "GPU INFO :: nvidia-smi"
nvidia-smi --query-gpu=name,compute_cap --format=csv

banner "RUNTIME ENV"
export CUDA_LOG_FILE="stdout"
echo "CUDA_LOG_FILE=$CUDA_LOG_FILE"

banner "RUN :: vanity-miner self-test"
./vanity-miner self-test
EOF
