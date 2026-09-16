# GPU kernel definitions

## Historical per-thread stack measurements

Rust-CUDA's NVVM backend inlines whole pipelines per kernel. The k256
(secp256k1) derive — used by `ethereum` and `bitcoin` — pulls
in deep nested `.func` calls (`ProjectivePoint::add`/`::double`,
`FieldElement::invert`, plus the SecretKey→Scalar→AffinePoint chain) that
together overflow CUDA's default per-thread stack of **1024 bytes** the
moment they're composed with a runtime priv input. Launch faults with
`illegal memory access`; `compute-sanitizer` is the only tool that names
the cause directly.

Empirically (RTX 4090, compute 8.9, CUDA 13.2): k256-derive kernels with
runtime priv need > 8 KiB and work at 16 KiB. Pipelines that don't compose
k256 (e.g. solana = xoroshiro + ed25519/dalek + base58) fit in the 1 KiB
default. The original 118-slot self-test ran clean at 16 KiB; RSA/P-256 checks use the larger host-configured stack.

These measurements predate the current pipeline. The CLI's current default is
64 KiB in `cli/src/runner/cuda/context.rs`; override with `STACK_SIZE=N`.
Validate stack requirements again when composing deeper GPU pipelines.
