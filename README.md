# ed25519-vanity-rs
Ed25519 vanity generator with CUDA in Rust

## How to use

```shell
# from vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04

# configure ssh key
ssh-keygen -t rsa -b 3072 -f ~/.ssh/id_rsa -N ""

# configure git
git config --global user.name "Brandon Ros"
git config --global user.email $EMAIL

# clone repo
git clone git@github.com:brandonros/ed25519-vanity-rs.git

# install system dependencies
apt update
apt install -y pkg-config libssl-dev zlib1g-dev

# install llvm v7
curl -sSf -L -O http://security.ubuntu.com/ubuntu/pool/universe/libf/libffi7/libffi7_3.3-5ubuntu1_amd64.deb && \
curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/llvm-7_7.0.1-12_amd64.deb && \
curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/llvm-7-dev_7.0.1-12_amd64.deb && \
curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/libllvm7_7.0.1-12_amd64.deb && \
curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/llvm-7-runtime_7.0.1-12_amd64.deb && \
apt-get install -y ./*.deb && \
ln -s /usr/bin/llvm-config-7 /usr/bin/llvm-config

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"

# build
export LLVM_CONFIG="llvm-config-7"
export LLVM_LINK_STATIC=1
export RUST_LOG=info
export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64/"
cargo build
```
