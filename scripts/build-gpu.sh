#!/usr/bin/env bash
# Reproduce the CI GPU build locally. Run inside the flake's devshell:
#
#   nix develop --command bash scripts/build-gpu.sh
#
# Does a full clean rebuild of (1) the cuda-oxide codegen backend and (2)
# vanity-miner with the backend loaded as a rustc plugin. RUSTFLAGS mirror
# what cargo-oxide injects internally — see crates/cargo-oxide/src/backend.rs
# `build_rustflags` in the cuda-oxide source.

set -euo pipefail

CUDA_OXIDE_DIR="${CUDA_OXIDE_DIR:-$HOME/cuda-oxide}"
BRANCH="${CUDA_OXIDE_BRANCH:-reproduce-errors}"
REPO_URL="https://github.com/brandonros/cuda-oxide.git"

banner() {
    echo ""
    echo "=================================================================="
    echo "==  $1"
    echo "=================================================================="
}

banner "clean :: cargo clean"
cargo clean

banner "fetch :: cuda-oxide ($BRANCH) -> $CUDA_OXIDE_DIR"
if [ -d "$CUDA_OXIDE_DIR" ]; then
    echo "already present, leaving as-is (set CUDA_OXIDE_DIR=... to override)"
else
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$CUDA_OXIDE_DIR"
fi
( cd "$CUDA_OXIDE_DIR" && git -C . log -1 --oneline )

# Mirror cargo-oxide's env setup. The codegen backend .so links against
# librustc_driver-<hash>.so which lives at $(rustc --print sysroot)/lib —
# rustc itself adds that to its dlopen search internally, but the
# backend build (cargo build for crates/rustc-codegen-cuda) needs it
# both at link time (LIBRARY_PATH) and at runtime (LD_LIBRARY_PATH).
RUSTC_SYSROOT="$(rustc --print sysroot)"
RUSTC_LIB="$RUSTC_SYSROOT/lib"
echo "rustc sysroot: $RUSTC_SYSROOT"

banner "build :: librustc_codegen_cuda.so"
# Not a workspace member — must cd into the crate directory and build standalone.
(
    cd "$CUDA_OXIDE_DIR/crates/rustc-codegen-cuda"
    LIBRARY_PATH="$RUSTC_LIB${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    LD_LIBRARY_PATH="$RUSTC_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        cargo build
)

# CARGO_TARGET_DIR (set by the flake's shellHook to the current dir's target)
# wins over per-crate target dirs, so the .so may land in either place.
if [ -n "${CARGO_TARGET_DIR:-}" ] && [ -f "$CARGO_TARGET_DIR/debug/librustc_codegen_cuda.so" ]; then
    BACKEND_SO="$CARGO_TARGET_DIR/debug/librustc_codegen_cuda.so"
elif [ -f "$CUDA_OXIDE_DIR/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so" ]; then
    BACKEND_SO="$CUDA_OXIDE_DIR/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so"
else
    echo "ERROR: librustc_codegen_cuda.so not found after build" >&2
    exit 1
fi
echo "backend: $BACKEND_SO"

banner "build :: vanity-miner (release, gpu) with NVPTX backend"
# Critical flags (mirrored from cargo-oxide's build_rustflags):
#   -Z always-encode-mir=yes      cross-crate pub fn calls into logic/ need MIR
#   -Z mir-enable-passes=-...     suppresses JumpThreading (barrier duplication)
#   -Csymbol-mangling-version=v0  matches what the backend expects
#   -C opt-level=3 / debug-assertions=off  required for sane PTX output
#
# LD_LIBRARY_PATH must include the rustc sysroot/lib so the backend .so
# (loaded into rustc via -Z codegen-backend) can resolve librustc_driver-*.so.
# This mirrors cargo-oxide's apply_ld_library_path helper.
unset CUDA_OXIDE_BACKEND
LD_LIBRARY_PATH="$RUSTC_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
RUSTFLAGS="-Z codegen-backend=$BACKEND_SO -C opt-level=3 -C debug-assertions=off -Z mir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0 -Z always-encode-mir=yes" \
    cargo build -p vanity-miner --features gpu --release

banner "result"
ls -lh target/release/vanity-miner
file target/release/vanity-miner
