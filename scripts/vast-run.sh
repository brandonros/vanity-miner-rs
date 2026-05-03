#!/usr/bin/env bash

set -e

PORT=13375
HOST=ssh9.vast.ai
USER=root

ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
VERSION="v1.18.0"

# check for killall
if ! command -v killall &> /dev/null
then
    apt update
    apt install -y psmisc
fi

# cleanup
rm -f vanity-miner
killall vanity-miner || true

# download the right arch (PTX is embedded — single binary)
ARCH=$(uname -m)  # x86_64 or aarch64
curl -fL -o vanity-miner https://github.com/brandonros/vanity-miner-rs/releases/download/$VERSION/vanity-miner-$ARCH
chmod +x vanity-miner

# run
export BLOCKS_PER_SM="1024"
export THREADS_PER_BLOCK="256"
export STACK_SIZE="8192"

#./vanity-miner solana-vanity aaaa ""
#./vanity-miner bitcoin-vanity bc1qqqqqq ""
#./vanity-miner ethereum-vanity 55555555 ""
./vanity-miner shallenge brandonros 0000000000bd0310ff0f88ac484f7fcd256ef78ae0deecd5693ed0ead124d17b
EOF
