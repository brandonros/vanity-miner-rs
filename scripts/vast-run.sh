#!/bin/bash

set -e

PORT=27992
HOST=ssh4.vast.ai
USER=root

ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
if [[ ! -f ed25519_vanity ]]
then
  curl -L -O https://github.com/brandonros/ed25519-vanity-rs/releases/download/1.1.0/ed25519_vanity
  chmod +x ed25519_vanity
fi
./ed25519_vanity Muffin ""
EOF
