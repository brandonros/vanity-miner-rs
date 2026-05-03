#!/usr/bin/env bash

set -e

PORT=15975
HOST=ssh1.vast.ai
USER=root

ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
VERSION="v1.23.0"

banner() {
    echo ""
    echo "=================================================================="
    echo "==  $1"
    echo "=================================================================="
}

banner "ENV CHECK :: killall"
if ! command -v killall &> /dev/null
then
    apt update
    apt install -y psmisc
else
    echo "killall already installed"
fi

banner "CLEANUP :: previous binary + running processes"
rm -f vanity-miner
killall vanity-miner || true

banner "DOWNLOAD :: vanity-miner $VERSION"
ARCH=$(uname -m)  # x86_64 or aarch64
echo "arch=$ARCH version=$VERSION"
curl -fL -o vanity-miner https://github.com/brandonros/vanity-miner-rs/releases/download/$VERSION/vanity-miner-$ARCH
chmod +x vanity-miner
ls -lh vanity-miner

banner "GPU INFO :: nvidia-smi"
nvidia-smi --query-gpu=name,compute_cap --format=csv

banner "RUNTIME ENV"
export CUDA_LOG_FILE="stdout"
export BLOCKS_PER_SM="1024"
export THREADS_PER_BLOCK="256"
export STACK_SIZE="8192"
echo "CUDA_LOG_FILE=$CUDA_LOG_FILE"
echo "BLOCKS_PER_SM=$BLOCKS_PER_SM"
echo "THREADS_PER_BLOCK=$THREADS_PER_BLOCK"
echo "STACK_SIZE=$STACK_SIZE"

banner "RUN :: vanity-miner"
./vanity-miner solana-vanity aaaa ""
#./vanity-miner bitcoin-vanity bc1qqqqqq ""
#./vanity-miner ethereum-vanity 55555555 ""
#./vanity-miner shallenge brandonros 0000000000bd0310ff0f88ac484f7fcd256ef78ae0deecd5693ed0ead124d17b
EOF
