#!/bin/bash

PORT=18960
HOST=ssh8.vast.ai
USER=root

# generate key
ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
if [[ ! -f ~/.ssh/id_rsa ]]
then
  ssh-keygen -t rsa -b 3072 -f ~/.ssh/id_rsa -N ""
  chmod 600 ~/.ssh/id_rsa
fi
EOF

# copy it
scp -P $PORT $USER@$HOST:~/.ssh/id_rsa.pub /tmp

# add it
gh ssh-key add /tmp/id_rsa.pub

# install dependencies
ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
if ! dpkg -l | grep -q "^ii  pkg-config "
then
  apt update
  apt install -y pkg-config libssl-dev zlib1g-dev clang
fi

if [[ ! -f /usr/bin/llvm-config-7 ]]
then
  curl -sSf -L -O http://security.ubuntu.com/ubuntu/pool/universe/libf/libffi7/libffi7_3.3-5ubuntu1_amd64.deb && \
  curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/llvm-7_7.0.1-12_amd64.deb && \
  curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/llvm-7-dev_7.0.1-12_amd64.deb && \
  curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/libllvm7_7.0.1-12_amd64.deb && \
  curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/llvm-7-runtime_7.0.1-12_amd64.deb && \
  apt-get install -y ./*.deb && \
  ln -s /usr/bin/llvm-config-7 /usr/bin/llvm-config && \
  rm ./*.deb
fi

if [[ ! -f ~/.cargo/env ]]
then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly-2025-03-02
fi
EOF

# build
ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
if [[ ! -d ed25519-vanity-rs ]]
then
  GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:brandonros/ed25519-vanity-rs.git
fi
pushd ed25519-vanity-rs
git fetch
git checkout --force no-compact
git reset --hard origin/no-compact
. $HOME/.cargo/env
export LLVM_CONFIG="llvm-config-7"
export LLVM_LINK_STATIC="1"
export RUST_LOG="info"
export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64/"
cargo build --release
EOF

# run
ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
pushd ed25519-vanity-rs
./target/release/ed25519_vanity aa 128 128
EOF
