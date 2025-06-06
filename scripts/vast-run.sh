#!/bin/bash

set -e

PORT=39920
HOST=ssh4.vast.ai
USER=root

ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
if [[ ! -f gpu_runner ]]
then
  curl -L -O https://github.com/brandonros/ed25519-vanity-rs/releases/download/1.2.0/gpu_runner
  chmod +x gpu_runner
fi
./gpu_runner shallenge brandonros 00000000002a6043ad35708ef4e05c7f8d64c5a0e74d71312886fdc4b41eee0f
EOF
