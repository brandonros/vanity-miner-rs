# CuMetal compatibility milestones

The target is numerical correctness for every exported miner kernel on Apple
GPU. A successful PTX build or MSL compilation is a separate milestone, not a
numerical pass. Keep the PTX unchanged throughout the consumer tests.

1. **Fresh full PTX artifact.** Run `cuda-compile.yaml` on this branch with
   `publish_release=false`. It builds all features on x86 Linux, verifies the
   complete source entry inventory (currently 123), and uploads
   `vanity-miner-x86_64`. This does not create a tag or release. The artifact
   contains `output.ptx`, `kernel-manifest.json`, `SHA256SUMS`, the source commit,
   Rust version, build command, and both Cargo lockfiles plus the flake lock.
2. **Compile inventory.** Feed each entry independently into CuMetal's strict
   typed backend. Record pass/failure and the first diagnostic for all 123.
   Group failures by opcode, ABI, memory, helper-call, or control-flow gap.
   Fix one group at a time with positive and negative regression coverage.
3. **Launch probe.** Run `kernel_self_test_stub`, verify its result and guards,
   and require actual generic-PTX Apple-GPU provenance.
4. **Primitive self-tests.** Start with SHA-256, SHA-512, Keccak, RIPEMD-160,
   xoroshiro, and base58; then tackle Ed25519 and secp256k1. Run individual
   kernels so a failure identifies a specific subsystem.
5. **All self-tests.** Execute the remaining composed-pipeline and compiler
   regression checks. Target: all 118 check kernels plus the plumbing probe,
   each with the expected result and intact guards.
6. **Four bounded mining workloads.** Validate Shallenge, Solana, Bitcoin, and
   Ethereum separately against CPU results with deterministic inputs and finite
   launches. Check reported matches, outputs, and bounds before any throughput
   measurements.

The final numerical denominator is 123 entry points: four mining kernels,
118 self-test checks, and one launch probe. Track not-run, compile failures,
launch failures, numerical mismatches, and passes separately. Passing one
bounded test per entry does not establish correctness for every possible input.

The existing miner LLVM 7 / `compute_89` toolchain remains in use. Building the
artifact does not require an NVIDIA GPU. The CLI build script forwards enabled
mining/self-test features explicitly into the separate kernel workspace;
`cargo build -p vanity-miner --all-features --release --locked` requests all five
kernel feature groups. A normal default build remains Shallenge-only.

Release publication remains an explicit workflow opt-in: `publish_release=true`
builds both host architectures, patches the binaries, and runs the tag/release
job. The default artifact-only binary retains its Nix runtime paths; the PTX
artifact is the input used for CuMetal compatibility testing.
