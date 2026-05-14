# Setup

This project is in a temporary state to debug a new "alpha" quality Rust -> CUDA compiler by Nvidia (NVLabs)

Relevant files focused on our "self test mode":

vanity-miner-rs
    cli/src/modes/self_test.rs
    logic/src/self_test.rs
    kernels/src/lib.rs

# Documentation

vanity-miner-rs/KNOWN_FAILURES.md is our scratchpad where we are tracking state of what is failing/passing as we try to hone in on errors from the compiler.

cuda-oxide/crates/rustc-codegen-cuda/examples/README.md is our scratchpad where we are tracking state f 

# Flow

In cuda-oxide project, under `crates/rustc-codegen-cuda/examples`, we recreate reproduction failure test cases that act like regression tests.
Then, we use test driven development (watch it fail, fix it, watch it pass).
Sometimes you get a compiler failure, sometimes you silently get a compiler success but the output .ll/.ptx has a bug that is not straight forward or clear. 
This is a deep hard subject of cross-compiler internals.
We are aiming to correlate each "vanity-minter-rs failing self test step" to at least 1 "cuda-oxide rustc-codegen-cuda" reproduction "test case" example.

# Goal

vanity-miner-rs self test passes entirely on the CPU. When it gets "cross-compiled" to CUDA through cuda-oxide, the kernels FAIL instead of PASS. The reason is not known. We want to understand the reason and create fixes in the cuda-oxide. We are on a massive branch in cuda-oxide that has 100+ commits/fixes as we work through these issues. We are also insanely devoted and do not care if it takes 200-300 commits/test cases. This entire exercise can be repurposed to become a great test case suite for GPUs.

## How to iterate

```shell
# do your fix in cuda-oxide, commit it and push it
# switch to vanity-miner-rs

# pull in latest rev git hash of https://github.com/brandonros/cuda-oxide.git reproduce-errors branch
cargo update -p cuda-core

# if needed, bump cargo.lock and commit it and push it
git add Cargo.lock
git commit -m "cuda-core bump"
git push

# spawn github actions workflow which will compile and publish github release
gh workflow run cuda-compile.yaml --ref cuda-oxide
sleep 5
gh run watch $(gh run list --workflow=cuda-compile.yaml --branch=cuda-oxide -L1 --json databaseId -q '.[0].databaseId') --exit-status

# run on a vast vm that has gpu and pulls the latest github release/tag
./scripts/vast-run.sh
```
