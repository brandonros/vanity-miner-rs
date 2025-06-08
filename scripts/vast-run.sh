#!/bin/bash

set -e

PORT=10178
HOST=ssh2.vast.ai
USER=root

ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
rm -f gpu_runner
if [[ ! -f gpu_runner ]]
then
  curl -L -O https://github.com/brandonros/ed25519-vanity-rs/releases/download/1.4.0/gpu_runner
  chmod +x gpu_runner
fi
export BLOCKS_PER_SM="1024"
export THREADS_PER_BLOCK="256"
killall gpu_runner || true
./gpu_runner shallenge brandonros 000000000000cbaec87e070a04c2eb90644e16f37aab655ccdf683fdda5a6f96
EOF
