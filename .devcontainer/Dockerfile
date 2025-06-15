FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    build-essential \
    clang \
    curl \
    libssl-dev \
    libtinfo-dev \
    pkg-config \
    xz-utils \
    zlib1g-dev \
    git \
    gdb \
    lldb \
    psmisc \
    libzstd-dev && \
    rm -rf /var/lib/apt/lists/*

# Get Rust
RUN curl -sSf -L https://sh.rustup.rs | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Add nvvm to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=info

# export PATH="/opt/llvm-19-debug/bin:$PATH"
