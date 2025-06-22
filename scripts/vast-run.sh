#!/bin/bash

set -e

PORT=25516
HOST=ssh3.vast.ai
USER=root

ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
VERSION=1.10.0

# cleanup
rm -f gpu_runner
rm -f output.cubin
killall gpu_runner || true

# download
curl -L -O https://github.com/brandonros/ed25519-vanity-rs/releases/download/$VERSION/gpu_runner
curl -L -O https://github.com/brandonros/ed25519-vanity-rs/releases/download/$VERSION/output.cubin
chmod +x gpu_runner

# run
export BLOCKS_PER_SM="1024"
export THREADS_PER_BLOCK="256"
export CUBIN_PATH="output.cubin"
./gpu_runner bitcoin-vanity bc1qqqqq ""
EOF
