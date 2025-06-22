FROM docker.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# set the path for the nvvm library
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"

# install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq -y install curl lsb-release wget software-properties-common gnupg git pkg-config libssl-dev zlib1g-dev libzstd-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install llvm 19
RUN curl -L -O https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 19

# install polly
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq -y install libpolly-19-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install rust
RUN curl -sSf -L https://sh.rustup.rs | bash -s -- -y --default-toolchain 1.86.0 --profile complete
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CARGO_TARGET_DIR="/root/.cargo/target"
RUN rustup target add riscv64gc-unknown-none-elf
