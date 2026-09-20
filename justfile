# Run inside the Nix shell: `nix develop --command just <recipe>`.
# Compiler development: add `--override-input llvm-metal path:../llvm-metal` to `nix develop`.

set shell := ["bash", "-euo", "pipefail", "-c"]

# List the recipes.
default:
    @just --list

# Build one kernel crate (a mode or self-test-<mode>) into target/metal/<mode>; options go to `llvm-metalc build`.
build mode *options:
    #!/usr/bin/env bash
    set -euo pipefail
    device=crates/kernels/{{mode}}
    target=$PWD/target/metal/device/{{mode}}
    cargo=(--locked --release --manifest-path "$device/Cargo.toml" --target-dir "$target")
    # A bundle is built only from a kernel whose CPU known answers pass.
    cargo test "${cargo[@]}"
    cases=()
    if [[ {{mode}} == self-test-* ]]; then
        # The host registry lists the cases; the compiler checks the device declares the same.
        cargo run "${cargo[@]}" --example inventory > "$target/cases.json"
        cases=(--cases "$target/cases.json")
    fi
    llvm-metalc build --crate "$device" --target-dir "$target" --output target/metal/{{mode}} \
        ${cases[@]+"${cases[@]}"} {{options}}

# Build every kernel crate.
build-all:
    for manifest in crates/kernels/*/Cargo.toml; do just build "$(basename "$(dirname "$manifest")")"; done

# Build every bundle, then run the GPU tests; a filter selects tests by name.
test *filter: build-all (test-built filter)

# Run the GPU tests against bundles that are already built. One thread: they share the device.
test-built *filter:
    cargo test --locked --release -p vanity-miner --all-features --test metal_gpu -- \
        --ignored --nocapture --test-threads=1 {{filter}}

# Build a mode's bundle and the host CLI, then run the CLI with the given arguments.
run mode *arguments: (build mode)
    cargo build --locked --release -p vanity-miner --no-default-features \
        --features "metal,$(m={{mode}}; [[ $m == self-test-* ]] && echo "${m//-/_}" || echo "$m")" \
        --target-dir target/metal/host
    target/metal/host/release/vanity-miner {{arguments}}
