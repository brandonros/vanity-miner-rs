//! solana self-tests: primitives, pipeline stages, and regressions.
pub(super) mod batch_seed_probes;
pub(super) mod candidate_probes;
mod scalar_fixtures;
use super::IdxProbe;
use super::bytes_eq_prefix;
use crate::crypto::ed25519::ed25519_derive_public_key;
use crate::crypto::sha512::sha512_32bytes_from_bytes;
use crate::encoding::base58::base58_encode;
use crate::encoding::base58::base58_encode_32;
use crate::modes::solana::SolanaVanityKeyRequest;
use crate::modes::solana::SolanaVanityKeyResult;
use crate::modes::solana::generate_and_check_solana_vanity_key;
use crate::search::xoroshiro::generate_random_private_key;

// === Solana per-primitive bisect (slots 0-3) ===
// The `solana priv` slot ran the *whole* pipeline before checking the priv
// bytes; if that kernel faulted we couldn't tell which primitive triggered
// it. These four `primitive_*` functions exercise each stage in
// isolation against externally-validated intermediates, so GPU mode can
// localize a fault to xoroshiro / sha512 / ed25519 / base58.

const SOLANA_PRIMITIVE_PRIV: [u8; 32] = [
    0xfa, 0x9c, 0xe9, 0xb0, 0x2d, 0xc2, 0x8a, 0x48, 0xf7, 0xe9, 0xd1, 0x55, 0x06, 0xd3, 0xd2, 0xc4,
    0x43, 0xd5, 0x96, 0x56, 0x5f, 0xa0, 0x52, 0x14, 0xb0, 0xff, 0x7c, 0x5a, 0xb5, 0xe7, 0x95, 0x6b,
];

const SOLANA_PRIMITIVE_HASHED_PRIV: [u8; 64] = [
    0xaa, 0xe4, 0x1d, 0x15, 0x43, 0x8a, 0x30, 0xa5, 0x0e, 0x27, 0x4b, 0x13, 0x6d, 0x5c, 0x2a, 0x7c,
    0x36, 0x6e, 0x68, 0xbf, 0xf9, 0xa0, 0xbb, 0x05, 0x87, 0x2c, 0x35, 0x75, 0x2e, 0x9a, 0x45, 0xa4,
    0x8c, 0x25, 0x5f, 0x21, 0xb8, 0x43, 0xfc, 0xa7, 0x21, 0x81, 0x3f, 0xc2, 0x40, 0x3e, 0x20, 0x13,
    0xe0, 0xe8, 0x1d, 0xd6, 0xd7, 0xc9, 0xd8, 0x69, 0xac, 0xf6, 0x03, 0x1e, 0x33, 0xb6, 0x95, 0x6a,
];

const SOLANA_PRIMITIVE_PUB: [u8; 32] = [
    0x08, 0x9a, 0x23, 0xff, 0xc4, 0x22, 0xf5, 0x3d, 0x11, 0x45, 0x87, 0x01, 0x2b, 0xb2, 0xc0, 0x28,
    0x49, 0x2f, 0xab, 0xda, 0xbe, 0x12, 0x66, 0xbc, 0x9a, 0xd6, 0x69, 0x8a, 0xc4, 0x30, 0x16, 0xbb,
];

register_self_test! {
    /// xoroshiro priv
    fn primitive_xoroshiro() -> u32 {
        let priv_key = generate_random_private_key(core::hint::black_box(3), core::hint::black_box(583437459223573146));
        (priv_key == SOLANA_PRIMITIVE_PRIV) as u32
    }
}

register_self_test! {
    /// sha512 of priv
    fn primitive_sha512() -> u32 {
        let hashed = sha512_32bytes_from_bytes(&core::hint::black_box(SOLANA_PRIMITIVE_PRIV));
        (hashed == SOLANA_PRIMITIVE_HASHED_PRIV) as u32
    }
}

register_self_test! {
    /// ed25519 derive
    fn primitive_ed25519() -> u32 {
        let pub_key = ed25519_derive_public_key(&core::hint::black_box(SOLANA_PRIMITIVE_HASHED_PRIV));
        (pub_key == SOLANA_PRIMITIVE_PUB) as u32
    }
}

register_self_test! {
    /// base58 encode pub
    fn primitive_base58() -> u32 {
        let expected: &[u8] = b"aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC";
        let mut out = [0u8; 64];
        let n = base58_encode_32(&core::hint::black_box(SOLANA_PRIMITIVE_PUB), &mut out);
        (n == expected.len() && bytes_eq_prefix(&out, expected)) as u32
    }
}

// === Solana (rng_seed=583437459223573146, thread_idx=3) ===

fn solana_test() -> SolanaVanityKeyResult {
    let req = SolanaVanityKeyRequest {
        prefix: b"",
        suffix: b"",
        thread_idx: 3,
        rng_seed: 583437459223573146,
    };
    generate_and_check_solana_vanity_key(&core::hint::black_box(req))
}

register_self_test! {
    /// solana priv
    fn private_key() -> u32 {
        let expected: [u8; 32] = [
            0xfa, 0x9c, 0xe9, 0xb0, 0x2d, 0xc2, 0x8a, 0x48, 0xf7, 0xe9, 0xd1, 0x55, 0x06, 0xd3, 0xd2,
            0xc4, 0x43, 0xd5, 0x96, 0x56, 0x5f, 0xa0, 0x52, 0x14, 0xb0, 0xff, 0x7c, 0x5a, 0xb5, 0xe7,
            0x95, 0x6b,
        ];
        (solana_test().private_key == expected) as u32
    }
}

register_self_test! {
    /// solana pub
    fn public_key() -> u32 {
        let expected: [u8; 32] = [
            0x08, 0x9a, 0x23, 0xff, 0xc4, 0x22, 0xf5, 0x3d, 0x11, 0x45, 0x87, 0x01, 0x2b, 0xb2, 0xc0,
            0x28, 0x49, 0x2f, 0xab, 0xda, 0xbe, 0x12, 0x66, 0xbc, 0x9a, 0xd6, 0x69, 0x8a, 0xc4, 0x30,
            0x16, 0xbb,
        ];
        (solana_test().public_key == expected) as u32
    }
}

register_self_test! {
    /// solana encoded
    fn encoded() -> u32 {
        let expected: &[u8] = b"aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC";
        let sol = solana_test();
        (sol.encoded_len == expected.len() && bytes_eq_prefix(&sol.encoded_public_key, expected)) as u32
    }
}

// === Arithmetic primitive bisect (slots 31-40) ===
// The composed primitives above all reduce to the same root cause: any
// integer op that lowers to `mul.hi.u64` (multi-word multiply, divide-by-
// constant via magic-multiply) returns wrong bytes on the current device.
// These slots pin down exactly which PTX op is broken so the alpha-compiler
// regression can be reported against a one-line repro.
//
// Pattern: each `arith_*` baselines the expected value via a `const`
// evaluated by the *host* rustc (correct, well-tested code), then runs the
// same expression at runtime with both operands hidden behind `black_box`
// so the GPU codegen can't constant-fold. Mismatch on GPU + match on CPU =
// codegen bug isolated to that op.

const ARITH_U32_A: u32 = 0xDEADBEEF;

const ARITH_U32_B: u32 = 0x12345678;

const ARITH_U64_A: u64 = 0xDEADBEEFCAFEBABE;

const ARITH_U64_B: u64 = 0x123456789ABCDEF0;

const ARITH_U128_A: u128 = ((ARITH_U64_A as u128) << 64) | (ARITH_U64_B as u128);

const ARITH_U128_B: u128 = ((ARITH_U64_B as u128) << 64) | (ARITH_U64_A as u128);

// === Composed-primitive sub-bisects (slots 41-45) ===

// 25-byte test vector that lives entirely in the divide-by-58 loop with no
// leading-zero pad. Exercises `base58_encode` (variable length) as opposed
// to `base58_encode_32` (fixed) already covered by slot 3.
const BASE58_VAR_INPUT: [u8; 25] = [
    0x0A, 0xF7, 0x64, 0xC1, 0xB6, 0x13, 0x3A, 0x3A, 0x0A, 0xBD, 0x7E, 0xF9, 0xC8, 0x53, 0x79, 0x1B,
    0x68, 0x7C, 0xE1, 0xE2, 0x35, 0xF9, 0xDC, 0x84, 0x66,
];

const BASE58_VAR_EXPECTED: &[u8] = b"5Qw8TAab98QrQmymczzxwkZzacMDL4MeEH";

// 32 all-zero bytes → all-leading-zero pad with no divide loop iterations.
// If this PASSes but slot 3 FAILs, the divide-by-58 codegen is to blame;
// if it FAILs the leading-zero pad logic itself is broken.
const BASE58_ALLZERO_EXPECTED: &[u8] = b"11111111111111111111111111111111";

// === Tier-2 arithmetic bisect (slots 46-56) ===
// Targets the PTX idioms heavily used by dalek/k256 that the tier-1 net
// (slots 31-40) doesn't directly exercise: carry-chain plumbing, both-lane
// extraction from a single widening mul, fused mul-add, 32×32→64, the
// subtle::Choice mask-blend pattern, and variable shifts.

// Slot 72: `EdwardsPoint::mul_base(scalar=1).compress()` must equal the
// well-known ed25519 basepoint encoding (RFC 8032). Exercises the full
// fixed-base scalar-mult path with the smallest non-trivial scalar.
//
// If slots 70/71 PASS and 72 FAILs, the bug is in mul_base / EdwardsPoint
// ops or in `compress()` (field inversion), not in the scalar plumbing.
const ED25519_BASEPOINT_COMPRESSED: [u8; 32] = [
    0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
];

// Slot 82: depth-4 newtype nesting on `&'static` data. k256's `Scalar::ONE`
// is a `pub const Scalar = Self(U256::ONE)` where:
//   Scalar(U256)
//     U256 = Uint<4> { limbs: [Limb; 4] }
//       Limb(u64)
// So accessing the inner u64 requires `scalar.0.limbs[i].0` — 4 levels of
// field projection. Slot 77 tested depth-2 (`Wrap([u64; 5])` + index).
// If THIS fails, the bug is GEP through nested newtypes, not array reads.
#[repr(transparent)]
pub struct ProbeLimb(pub u64);

#[repr(C)]
pub struct ProbeUint4 {
    pub limbs: [ProbeLimb; 4],
}

#[repr(transparent)]
pub struct ProbeScalar(pub ProbeUint4);

static NESTED_ONE_PROBE: ProbeScalar = ProbeScalar(ProbeUint4 {
    limbs: [
        ProbeLimb(0x1111_2222_3333_4444),
        ProbeLimb(0x5555_6666_7777_8888),
        ProbeLimb(0x9999_AAAA_BBBB_CCCC),
        ProbeLimb(0xDDDD_EEEE_FFFF_0000),
    ],
});

// === Slots 84-87: ladder bisect of slot 71's call chain ===
//
// Slot 71's `Scalar::from_bytes_mod_order(x).to_bytes()` for x=1 expands
// inside dalek to:
//   1. Scalar52::from_bytes(&bytes)            — byte→limb unpack
//   2. Scalar52::mul_internal(x, R)            — 5×5 widening mul matrix
//   3. Scalar52::montgomery_reduce(xR)         — u128 chain + L reads
//   4. result.as_bytes()                       — limb→byte pack
//
// dalek's `backend` module is `pub(crate)`, so we can't reach Scalar52,
// mul_internal, montgomery_reduce, or constants::L/R from outside. To
// ladder-bisect we copy the minimum needed from dalek into this module
// verbatim — same Rust source, just compiled inside the logic crate so
// each step is callable in isolation.
//
// All names prefixed `bisect_` to make it obvious these are not the real
// dalek types, even though the code is byte-for-byte identical to
// `curve25519-dalek/src/backend/serial/u64/scalar.rs`.
mod bisect_scalar52;

const DALEK_ONE_LIMBS: [u64; 5] = [1, 0, 0, 0, 0];

// Slot 91: focused Index/IndexMut trait dispatch probe on a tuple
// struct. Mirrors dalek's Scalar52 Index impl shape EXACTLY: tuple
// struct wrapping `[u64; 5]`, Index returns `&u64`, IndexMut returns
// `&mut u64`. If this FAILs, trait dispatch on `[i]` syntax is broken
// on the cuda-oxide alpha-NVPTX backend — explains why dalek (uses
// `a[i]`) fails while our port (uses `a.0[i]`) passes.

// Slot 106: Named-field struct wrapping `[u8; 32]` return-by-value test.
// This is the EXACT shape of dalek's `Scalar`:
//   pub struct Scalar { pub(crate) bytes: [u8; 32] }
// All known-passing return shapes:
//   - `clamp_integer` returns `[u8; 32]` direct (slot 70 PASS)
//   - `Scalar52::from_bytes` returns tuple-struct `Scalar52(pub [u64; 5])` (slot 84 PASS)
//   - `Scalar::ONE.to_bytes()` returns `[u8; 32]` direct (slot 92 PASS)
// All known-failing through dalek's Scalar:
//   - `Scalar::from_bytes_mod_order` returns Scalar (named-field struct) (slot 71/102/103 FAIL)
// Hypothesis: returning a named-field struct wrapping `[u8; 32]` by
// value is miscompiled. If 106 FAILs, that's the minimum Bug-71 repro.
#[repr(C)]
pub struct WrapNamed {
    pub bytes: [u8; 32],
}

#[inline(never)]
fn make_wrap_named(input: [u8; 32]) -> WrapNamed {
    let mut bytes = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        bytes[i] = input[i].wrapping_add(1);
        i += 1;
    }
    WrapNamed { bytes }
}

pub(super) mod base58_probes;

pub(super) mod ed25519_probes;

pub(super) mod layout_probes;

pub(super) mod arithmetic;
