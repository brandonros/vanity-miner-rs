#!/bin/bash

PORT=13360
HOST=ssh4.vast.ai
USER=root

ssh -p $PORT $USER@$HOST

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

# add it to github
gh ssh-key add /tmp/id_rsa.pub

# install dependencies
ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
if ! which pkg-config >/dev/null 2>&1; then
  apt update
  apt install -y pkg-config libssl-dev zlib1g-dev clang
fi

# llvm-7
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

# rust
if [[ ! -f ~/.cargo/env ]]
then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly-2025-03-02
fi
EOF

# build
ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
# clone
if [[ ! -d ed25519-vanity-rs ]]
then
  GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:brandonros/ed25519-vanity-rs.git
fi

# checkout
pushd ed25519-vanity-rs
git fetch
git checkout --force master
git reset --hard origin/master

# build
. $HOME/.cargo/env
cargo build --release
EOF

# run
ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
pushd ed25519-vanity-rs
./target/release/ed25519_vanity aa 8192 256
EOF
