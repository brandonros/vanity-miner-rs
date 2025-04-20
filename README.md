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

# install system dependencies
apt update
apt-get remove --purge 'cuda-12-0*' 'cuda-*12-0*' 'libcublas-12-0' 'libcublas-dev-12-0' cuda-12.0
apt-get autoremove
apt install pkg-config libssl-dev llvm-7 llvm-7-dev llvm-7-tools clang-7 zlib1g-dev cuda-12-8

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
