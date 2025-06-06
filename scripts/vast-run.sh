#!/bin/bash

set -e

PORT=39920
HOST=ssh4.vast.ai
USER=root

ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
if [[ ! -f gpu_runner ]]
then
  curl -L -O https://github.com/brandonros/ed25519-vanity-rs/releases/download/1.3.0/gpu_runner
  chmod +x gpu_runner
fi
export BLOCKS_PER_SM="1024"
export THREADS_PER_BLOCK="256"
killall gpu_runner
./gpu_runner shallenge brandonros 00000000000343368f85da942374a09773be369d7603b0370abee5388f2ea845
EOF
