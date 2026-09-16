#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [7|21]"
  echo "Compile all PTX kernels and self-tests with Nix (default: LLVM 21)."
  echo "Requires Linux and Nix with flakes enabled; no GPU is needed."
}

if [[ $# -eq 1 && ( $1 == --help || $1 == -h ) ]]; then
  usage
  exit 0
fi
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
llvm=${1:-21}
case "$llvm" in
  7|21) ;;
  *) usage >&2; exit 2 ;;
esac

if [[ $(uname -s) != Linux ]]; then
  echo "Run this script on Linux, for example with limactl shell vanity-nixos." >&2
  exit 1
fi
if ! command -v nix >/dev/null 2>&1; then
  echo "Nix is required (with nix-command and flakes enabled)." >&2
  exit 1
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

features=""
if [[ $llvm == 21 ]]; then
  features=llvm21
fi

modes=(solana bitcoin ethereum shallenge rsa-modulus rsa-pss p256-public-key p256-signature)
packages=()
ptx_files=()
for mode in "${modes[@]}"; do
  packages+=(-p "kernel-$mode" -p "kernel-self-test-$mode")
  module=${mode//-/_}
  ptx_files+=("$module.ptx" "self_test_$module.ptx")
done

# Each package runs its own build.rs without compiling the CLI.
# The devshell sets CARGO_TARGET_DIR to target/llvm<version>.
nix develop ".#v${llvm}" --command cargo check "${packages[@]}" \
  --features "$features" --release --locked

ptx_dir="target/llvm${llvm}/release/ptx"
for file in "${ptx_files[@]}"; do
  if [[ ! -s "$ptx_dir/$file" ]]; then
    echo "Missing PTX file: $ptx_dir/$file" >&2
    exit 1
  fi
done

mkdir -p artifacts
bundle="artifacts/ptx-bundle-llvm${llvm}.tar.gz"
# List the expected outputs explicitly, excluding stale or diagnostic modules.
tar -C "$ptx_dir" -czf "$bundle" "${ptx_files[@]}"
echo "PTX bundle: $repo_root/$bundle"
