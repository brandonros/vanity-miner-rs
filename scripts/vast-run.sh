#!/bin/bash

set -e

PORT=28770
HOST=ssh8.vast.ai
USER=root

scp -P $PORT nvvm_compiler/build/output.cubin $USER@$HOST:

ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
VERSION=1.12.0

# check for killall
if ! command -v killall &> /dev/null
then
    apt update
    apt install -y psmisc
fi

# cleanup
rm -f gpu_runner
#rm -f output.cubin
killall gpu_runner || true

# download
curl -L -O https://github.com/brandonros/ed25519-vanity-rs/releases/download/$VERSION/gpu_runner
#curl -L -O https://github.com/brandonros/ed25519-vanity-rs/releases/download/$VERSION/output.cubin
chmod +x gpu_runner

# run
export BLOCKS_PER_SM="256"
export THREADS_PER_BLOCK="256"
export STACK_SIZE="8192"
export CUBIN_PATH="output.cubin"
#./gpu_runner bitcoin-vanity bc1qqqqqq ""
./gpu_runner ethereum-vanity 55555555 ""
EOF
