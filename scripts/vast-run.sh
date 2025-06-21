#!/bin/bash

set -e

PORT=13580
HOST=ssh4.vast.ai
USER=root
VERSION=1.8.0

ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
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
./gpu_runner solana-vanity aaa ""
EOF