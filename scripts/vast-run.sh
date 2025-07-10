#!/bin/bash

set -e

PORT=18756
HOST=ssh6.vast.ai
USER=root

ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
VERSION="v1.14.0"

# check for killall
if ! command -v killall &> /dev/null
then
    apt update
    apt install -y psmisc
fi

# cleanup
rm -f gpu_runner
rm -f output.cubin
killall gpu_runner || true

# download
curl -L -O https://github.com/brandonros/vanity-miner-rs/releases/download/$VERSION/gpu_runner
curl -L -O https://github.com/brandonros/vanity-miner-rs/releases/download/$VERSION/output.cubin
chmod +x gpu_runner

# run
export BLOCKS_PER_SM="1024"
export THREADS_PER_BLOCK="256"
export STACK_SIZE="8192"
export CUBIN_PATH="output.cubin"

#./gpu_runner solana-vanity aaaa ""
#./gpu_runner bitcoin-vanity bc1qqqqqq ""
#./gpu_runner ethereum-vanity 55555555 ""
./gpu_runner shallenge brandonros 0000000000bd0310ff0f88ac484f7fcd256ef78ae0deecd5693ed0ead124d17b
EOF
