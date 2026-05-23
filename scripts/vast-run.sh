#!/usr/bin/env bash

set -e

PORT=39101
HOST=ssh5.vast.ai
USER=root

LOCAL_BINARY="target/release/vanity-miner"

if [ ! -f "$LOCAL_BINARY" ]; then
    echo "ERROR: $LOCAL_BINARY not found. Build it first with: cargo build --release --features gpu --features shallenge"
    exit 1
fi

ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
banner() {
    echo ""
    echo "=================================================================="
    echo "==  $1"
    echo "=================================================================="
}

banner "ENV CHECK :: killall + patchelf"
PKGS=""
command -v killall &> /dev/null || PKGS="$PKGS psmisc"
command -v patchelf &> /dev/null || PKGS="$PKGS patchelf"
if [ -n "$PKGS" ]; then
    apt update
    apt install -y $PKGS
else
    echo "killall + patchelf already installed"
fi

banner "CLEANUP :: previous binary + running processes"
rm -f vanity-miner
killall vanity-miner || true
EOF

banner_local() {
    echo ""
    echo "=================================================================="
    echo "==  $1"
    echo "=================================================================="
}

banner_local "UPLOAD :: scp local $LOCAL_BINARY -> $HOST:vanity-miner"
scp -o StrictHostKeyChecking=no -P $PORT "$LOCAL_BINARY" $USER@$HOST:vanity-miner

ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
banner() {
    echo ""
    echo "=================================================================="
    echo "==  $1"
    echo "=================================================================="
}

banner "PREPARE :: uploaded binary"
chmod +x vanity-miner
ls -lh vanity-miner

banner "PATCHELF :: fix nix interpreter -> system ld-linux"
# Binary was built in a nix shell, so its ELF interpreter points at a nix store
# path that doesn't exist on this host. Repoint at the system loader and drop
# the nix rpath so it links against system libs.
SYSTEM_LD=$(ls /lib64/ld-linux-x86-64.so.2 /lib/ld-linux-x86-64.so.2 2>/dev/null | head -n1)
echo "interpreter before: $(patchelf --print-interpreter vanity-miner)"
patchelf --set-interpreter "$SYSTEM_LD" vanity-miner
patchelf --remove-rpath vanity-miner
echo "interpreter after:  $(patchelf --print-interpreter vanity-miner)"

# Download path (kept for reference, replaced by scp upload above)
#VERSION="v1.24.0"
#ARCH=$(uname -m)  # x86_64 or aarch64
#echo "arch=$ARCH version=$VERSION"
#curl -fL -o vanity-miner https://github.com/brandonros/vanity-miner-rs/releases/download/$VERSION/vanity-miner-$ARCH

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
#./vanity-miner solana-vanity aaaa ""
#./vanity-miner bitcoin-vanity bc1qqqqqq ""
#./vanity-miner ethereum-vanity 55555555 ""
./vanity-miner shallenge brandonros FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
EOF
