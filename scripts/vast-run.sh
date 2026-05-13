#!/usr/bin/env bash

set -e

PORT=21385
HOST=ssh9.vast.ai
USER=root
VERSION="${VANITY_MINER_VERSION:-v1.29.0}"

ssh -o StrictHostKeyChecking=no -p "$PORT" "$USER@$HOST" <<EOF
banner() {
    echo ""
    echo "=================================================================="
    echo "==  \$1"
    echo "=================================================================="
}

banner "ENV CHECK :: killall"
if ! command -v killall &> /dev/null; then
    apt update && apt install -y psmisc
else
    echo "killall already installed"
fi

banner "CLEANUP :: stop any running miner + previous binary"
killall vanity-miner || true
rm -f vanity-miner kernels.ptx

banner "DOWNLOAD :: vanity-miner $VERSION (x86_64) + kernels.ptx"
curl -fL -o vanity-miner \
    https://github.com/brandonros/vanity-miner-rs/releases/download/$VERSION/vanity-miner-x86_64
curl -fL -o kernels.ptx \
    https://github.com/brandonros/vanity-miner-rs/releases/download/$VERSION/kernels.ptx
chmod +x vanity-miner
ls -lh vanity-miner kernels.ptx

banner "GPU INFO :: nvidia-smi"
nvidia-smi --query-gpu=name,compute_cap --format=csv

banner "RUNTIME ENV"
export CUDA_LOG_FILE="stdout"
# PTX_PATH tells the runner to load kernels from this file instead of the
# (currently-broken) embedded .oxart ELF section.
export PTX_PATH="\$PWD/kernels.ptx"
# Bump per-thread stack from the CUDA default (~1 KiB) so the self-test
# kernels' heaviest primitive (ed25519 scalar mult inside the solana check)
# has room. GpuContext::new picks this up and calls cuCtxSetLimit.
export STACK_SIZE=65536
echo "CUDA_LOG_FILE=\$CUDA_LOG_FILE"
echo "PTX_PATH=\$PTX_PATH"
echo "STACK_SIZE=\$STACK_SIZE"

banner "RUN :: vanity-miner self-test"
compute-sanitizer --tool memcheck --print-limit 0 --show-backtrace device ./vanity-miner self-test
EOF
