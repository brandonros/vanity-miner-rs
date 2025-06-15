FROM docker.io/nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

# install base dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq -y install curl lsb-release wget software-properties-common gnupg

# install llvm 19
RUN curl -L -O https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 19

# install layer 2 depdencncies
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install git pkg-config libssl-dev libpolly-19-dev zlib1g-dev libzstd-dev

# install rust
RUN curl -sSf -L https://sh.rustup.rs | bash -s -- -y --default-toolchain nightly-2025-03-02 --profile complete
ENV PATH="/root/.cargo/bin:${PATH}"

# set compliation flags
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_LINK_STATIC=1

