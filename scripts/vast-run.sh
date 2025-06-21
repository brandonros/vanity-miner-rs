#!/bin/bash

set -e

PORT=35954
HOST=ssh8.vast.ai
USER=root

ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
VERSION=1.9.0

rm -f gpu_runner
if [[ ! -f gpu_runner ]]
then
  curl -L -O https://github.com/brandonros/ed25519-vanity-rs/releases/download/$VERSION/gpu_runner
  curl -L -O https://github.com/brandonros/ed25519-vanity-rs/releases/download/$VERSION/kernels.ptx
  chmod +x gpu_runner
fi
export BLOCKS_PER_SM="1024"
export THREADS_PER_BLOCK="256"
export PTX_PATH="kernels.ptx"
killall gpu_runner || true
./gpu_runner add
EOF
