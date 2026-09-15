//! On-device / on-CPU self-test: runs known-answer tests for every logic
//! primitive against externally-validated expected values, writing
//! pass(1)/fail(0) per check into the results buffer.
//!
//! Each slot has a dedicated `check_*` function. GPU mode runs eight mode-specific
//! kernels, preserving individual result slots. CPU mode calls all checks in sequence.
//!
//! Keep known-answer inputs opaque before the operation under test. A barrier
//! around the final boolean is too late: the operation can already be folded.
//! `black_box` is best effort; inspect emitted PTX to verify the computation
//! survives optimization.

#[cfg(any(
    feature = "self_test_p256_public_key",
    feature = "self_test_p256_signature",
    feature = "self_test_rsa_pss",
    feature = "self_test_rsa_modulus"
))]
mod fixtures;
#[cfg(feature = "self_test_solana")]
mod solana;
#[cfg(feature = "self_test_solana")]
pub use solana::*;
#[cfg(feature = "self_test_bitcoin")]
mod bitcoin;
#[cfg(feature = "self_test_bitcoin")]
pub use bitcoin::*;
#[cfg(feature = "self_test_ethereum")]
mod ethereum;
#[cfg(feature = "self_test_ethereum")]
pub use ethereum::*;
#[cfg(feature = "self_test_shallenge")]
mod shallenge;
#[cfg(feature = "self_test_shallenge")]
pub use shallenge::*;
#[cfg(feature = "self_test_p256_public_key")]
mod p256_public_key;
#[cfg(feature = "self_test_p256_public_key")]
pub use p256_public_key::*;
#[cfg(feature = "self_test_p256_signature")]
mod p256_signature;
#[cfg(feature = "self_test_p256_signature")]
pub use p256_signature::*;
#[cfg(feature = "self_test_rsa_pss")]
mod rsa_pss;
#[cfg(feature = "self_test_rsa_pss")]
pub use rsa_pss::*;
#[cfg(feature = "self_test_rsa_modulus")]
mod rsa_modulus;
#[cfg(feature = "self_test_rsa_modulus")]
pub use rsa_modulus::*;

#[cfg(any(
    feature = "self_test_solana",
    feature = "self_test_bitcoin",
    feature = "self_test_ethereum"
))]
pub(super) fn bytes_eq_prefix(actual: &[u8; 64], expected: &[u8]) -> bool {
    let n = expected.len();
    let mut i = 0;
    while i < n {
        if actual[i] != expected[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(any(
    feature = "self_test_p256_public_key",
    feature = "self_test_p256_signature",
    feature = "self_test_rsa_pss",
    feature = "self_test_rsa_modulus"
))]
fn record_candidate(
    h: &mut crate::crypto::sha256::Sha256,
    result: crate::search::candidate_result::CandidateResult,
) {
    h.update(result.status.to_le_bytes());
    h.update(result.bytes);
}

#[cfg(any(feature = "self_test_solana", feature = "self_test_bitcoin"))]
pub struct IdxProbe(pub [u64; 5]);

#[cfg(any(feature = "self_test_solana", feature = "self_test_bitcoin"))]
impl core::ops::Index<usize> for IdxProbe {
    type Output = u64;
    fn index(&self, i: usize) -> &u64 {
        &(self.0[i])
    }
}

#[cfg(any(feature = "self_test_solana", feature = "self_test_bitcoin"))]
impl core::ops::IndexMut<usize> for IdxProbe {
    fn index_mut(&mut self, i: usize) -> &mut u64 {
        &mut (self.0[i])
    }
}

pub const SELF_TEST_NUM_CHECKS: usize = 157;

/// Slot labels in order; useful for printing results.
///
/// Slots 0–3 isolate the four primitives that the solana pipeline composes
/// (xoroshiro → sha512 → ed25519 → base58). Slots 4–9 isolate the remaining
/// primitives used by the bitcoin / ethereum / shallenge / WIF pipelines
/// (secp256k1 compressed + uncompressed, keccak256, ripemd160, and the two
/// sha256 entry points). Running all primitives first means a fault inside
/// any one surfaces in its own slot before the composed pipeline kernels
/// (slots 10+) would have inlined it.
pub const SELF_TEST_LABELS: [&str; SELF_TEST_NUM_CHECKS] = [
    "xoroshiro priv",
    "sha512 of priv",
    "ed25519 derive",
    "base58 encode pub",
    "secp256k1 compressed",
    "secp256k1 uncompressed",
    "keccak256 64bytes",
    "ripemd160 32bytes",
    "sha256 32bytes",
    "sha256 variable",
    "solana priv",
    "solana pub",
    "solana encoded",
    "ethereum priv",
    "ethereum pub",
    "ethereum address",
    "bitcoin priv",
    "bitcoin pub",
    "bitcoin pkh",
    "bitcoin encoded",
    "bitcoin matches",
    "wif compressed mainnet",
    "wif uncompressed mainnet",
    "wif compressed testnet",
    "wif uncompressed testnet",
    "shallenge hash",
    "shallenge nonce_len",
    "shallenge is_better",
    "compare_hashes lt",
    "compare_hashes gt",
    "compare_hashes eq",
    // Slots 31-40: micro-bisect of raw integer ops. The failing primitives
    // above (ed25519/secp256k1 field math, base58 divide-by-58) all use
    // `mul.hi.u64` in the emitted PTX; the passing ones don't. These slots
    // isolate each suspect op against a const-evaluated host-side baseline
    // so a per-op compiler bug surfaces directly.
    "arith u32 div var",
    "arith u32 div const",
    "arith u64 div var",
    "arith u64 div const",
    "arith u32 rem var",
    "arith u64 rem var",
    "arith u32 mul lo",
    "arith u64 mul lo",
    "arith u64 mul hi",
    "arith u128 mul",
    // Slots 41-45: composed-primitive sub-bisects that the broader checks
    // can't cleanly isolate.
    //   41 — base58 variable-length, no leading zeros (different entry
    //        point than base58_encode_32 at slot 3; same divide-by-58 loop)
    //   42 — base58 variable-length, leading-zero pad (Bitcoin P2PKH genesis)
    //   43 — base58 all-zero 32-byte input (pure leading-zero path, no
    //        numeric work — separates the divide loop from the pad logic)
    //   44 — xoroshiro base64 nonce (different code path from
    //        generate_random_private_key — next_u32 + alphabet lookup)
    //   45 — bech32 p2wpkh address encoding (only otherwise reached through
    //        the bitcoin composed kernel; isolating it tells us whether the
    //        bitcoin failure stops upstream at secp256k1 or also breaks here)
    "base58 var-len",
    "base58 var-len leading-zero",
    "base58 32 all-zeros",
    "xoroshiro base64 nonce",
    "bech32 p2wpkh",
    // Slots 46-56: second-tier arithmetic bisect targeting PTX idioms
    // dalek/k256 hit that the slot 31-40 net misses. Highest-signal entries:
    //   46-48 — carry-chain plumbing (add.cc.u64/addc.cc.u64). If broken,
    //           every multi-limb add silently corrupts bits → explains all
    //           three failing primitives in one stroke.
    //   53-54 — subtle::Choice-style mask blend. If broken, k256's
    //           SecretKey::from_bytes(...).unwrap() returns the wrong arm
    //           silently → matches "consistent but wrong" secp256k1.
    "arith overflowing_add",
    "arith overflowing_sub",
    "arith carry-chain 3limb",
    "arith widening mul pair",
    "arith mad lo u64",
    "arith mad hi u64",
    "arith mul wide u32",
    "arith mask blend true",
    "arith mask blend false",
    "arith var shr u64",
    "arith var shl u64",
    // Slots 57-58: smoking-gun probe for `core::hint::black_box`. If
    // black_box itself doesn't preserve its input on the device, every
    // tier-1 / tier-2 arith slot above is testing 0-vs-EXPECTED instead of
    // the intended operands → uniform FAIL that has nothing to do with the
    // op being probed. These two slots are the cheapest possible test:
    // identity check, no arithmetic at all.
    "arith blackbox identity u64",
    "arith blackbox identity u32",
    // Slot 59: confirms slot 41's crash is downstream of broken div/mod-
    // by-58 codegen (same mul.hi.u64 class as slot 40), not an independent
    // bug. Runs the exact divmod-by-58 pattern from base58_encode_32 in
    // isolation — no alphabet lookup, no output[] writes, no dynamic
    // loops — and compares against a host-const-eval baseline.
    "base58 div by 58",
    // Slots 60-62: triangulating bisects for the slot 41/43 fault class.
    // Slot 43 ([0u8; 32] → all-zero input) crashes despite the divide loop
    // being dynamically dead, so the shared crash path between 41 and 43
    // can't be the divide-by-58 codegen. Suspects narrow to the *final*
    // base58 stage: `for val in &mut output[..result_len]` IterMut over a
    // sub-slice of a stack-resident fixed-size array, and the dynamic
    // `TABLE[byte as usize]` lookup against a static byte alphabet. These
    // slots isolate each sub-pattern independently:
    //   60 — bare static-table dynamic lookup (no iter, no slice)
    //   61 — iter_mut over `&mut [u8; N][..n]` (no table lookup)
    //   62 — combined (mirrors the slot 43 final-loop shape exactly)
    "iter static table lookup",
    "iter mut slice partial",
    "iter mut alphabet lookup",
    // Slot 63: direct counterpart to slot 60. Slot 60 uses `&'static
    // [u8; N]` (array reference) for the table; this one uses `&'static
    // [u8]` (slice) — the *only* difference. Across the run history:
    // every crashing alphabet-lookup site uses array-ref; the one site
    // that uses a slice (xoroshiro/slot 44) FAILs without crashing. If 60
    // CRASHes and 63 PASSes, the array-ref-vs-slice discriminator is
    // confirmed and the alpha-NVPTX bug is narrowed to one Rust idiom.
    "iter static slice lookup",
    // Slots 64-65: targeted regressions ported from the cuda-oxide
    // standalone repros (divrem_large_const, i128_add_carry_chain). Each
    // mirrors the standalone kernel's logic but runs in this suite so
    // every future compiler bump re-validates them automatically.
    //   64 — `x / 58^5` and `x % 58^5` for the same divisor base58
    //        actually uses (NEXT_LIMB_DIVISOR = 656_356_768). Slot 59
    //        only goes up to /58^4; slot 64 covers the gap.
    //   65 — sequential `u128 + u128` wrapping chain that forces a
    //        low→high carry on every addition. Mirrors the
    //        accumulation pattern in dalek's Scalar52::mul_internal and
    //        k256's FieldElement5x52::mul_inner.
    "arith divrem by 58^5",
    "arith i128 chain add",
    // Slots 66-68: ported from cuda-oxide standalone repros that
    // existing slots don't cover.
    //   66 — `mul.hi.u64` with multiplicand reconstructed via
    //        `shl + add` from a stack-resident u32 limb (base58's
    //        exact inner-loop shape). Slot 64 feeds the divrem from
    //        a black_box'd u64 — different operand path. If 64
    //        passes and 66 fails, the bug is specifically `mul.hi.u64`
    //        with arithmetic-derived operands.
    //   67 — dynamic-grow stack-array writes:
    //        `while c > 0 { limbs[limb_count] = ...; limb_count += 1; }`.
    //        Distinct from slots 60-62 (read-side) — those use a
    //        static index variable / IterMut. This is the write side
    //        with the index variable mutating across iterations.
    //   68 — dalek/k256 partial-product accumulation:
    //        `a0*b2 + a1*b1 + a2*b0` via widening mul + u128 add.
    //        Slot 49 covers one widening pair; slot 65 covers a
    //        u128 add chain. This composes them — the actual shape
    //        Scalar52::mul_internal and FieldElement5x52::mul_inner
    //        emit.
    "base58 limb divrem (shl+add multiplicand)",
    "dynamic-grow stack array write",
    "widening mul chain 3-term",
    // Slot 69: Phase A of base58_encode_32's outer-loop body — the
    // in-place mutate over `for i in 0..limb_count` that reads each
    // existing limb, shifts+adds the carry, divrems by 58^5, and
    // writes back. Slot 67 covers Phase B (append-grow) but not this.
    // Slot 41 (non-zero base58 input) exercises Phase A starting at
    // outer iter 1; slot 43 (all-zero input) never enters Phase A
    // because limb_count stays 0 forever. That asymmetry points
    // squarely at Phase A.
    "base58 inner-mutate phase A",
    // Slots 70-72: ed25519 (curve25519-dalek) per-stage bisect.
    // Slot 2 (full ed25519_derive) fails; these isolate WHICH stage:
    //   70 — clamp_integer (pure bit-mask on byte 0 + byte 31, no arith).
    //        If FAIL, the most trivial dalek call is broken → look at
    //        public-API entry overhead, not the arithmetic.
    //   71 — Scalar::from_bytes_mod_order round-trip for the canonical
    //        scalar 1. Tests Scalar52 deserialization with no field math.
    //   72 — EdwardsPoint::mul_base(scalar=1).compress() must equal the
    //        well-known basepoint encoding. Tests the FULL fixed-base
    //        scalar-mult path with the smallest non-trivial scalar.
    "dalek clamp_integer",
    "dalek scalar from-bytes round-trip",
    "dalek mul_base scalar=1 == basepoint",
    // Slots 73-75: secp256k1 (k256) per-stage bisect for slot 4.
    //   73 — SecretKey::from_bytes for scalar=1: just the validation +
    //        wrap step. If FAIL alone, scalar reduction / Choice mask
    //        blend (subtle) is the suspect.
    //   74 — full derive for scalar=1: compressed pub must equal the
    //        well-known generator encoding. Smallest non-trivial input
    //        through the entire k256 pipeline.
    //   75 — full derive for scalar=2: must equal 2G compressed (also
    //        well-known). Exercises exactly one point doubling beyond
    //        slot 74; a 74-PASS / 75-FAIL split pinpoints doubling.
    "k256 SecretKey::from_bytes(1)",
    "k256 derive scalar=1 == generator",
    "k256 derive scalar=2 == 2G",
    // Slots 76-77: unifying-hypothesis probe.
    //   Slot 71 FAILs on `Scalar::from_bytes_mod_order(1).to_bytes()`.
    //   The only static read in that path is `Scalar52::reduce()`
    //   indexing `L: Scalar52([u64; 5])` — a `&'static` of u64s. Earlier
    //   runs proved `&'static [u8; N]` works (slots 60/63 PASS). So the
    //   discriminator might be element width, not array-vs-slice.
    //   76 — bare `&'static [u64; 5]` runtime-indexed read, no struct
    //        wrapper, no field math.
    //   77 — `&'static StructWrap([u64; 5])` indexed via `.0[i]` to
    //        match dalek/k256's actual newtype shape.
    "static [u64; 5] indexed read",
    "static struct-wrapped [u64; 5] indexed read",
    // Slots 78-80: k256 bug-triangulation probes (Bug B in KNOWN_FAILURES).
    // Slots 4/5/74/75 fail but k256's code path differs from dalek (uses
    // `Lazy<>` instead of `&'static const`), so these isolate WHICH
    // k256 subsystem is broken.
    //   78 — Encode generator directly (no scalar mult, no Lazy<>).
    //        FAIL → bug in projective→affine + encoding.
    //   79 — Double generator + encode (one doubling, no scalar mult).
    //        78 PASS / 79 FAIL → doubling formula or 5-wide field-mul
    //        carry chain.
    //   80 — k256 `Scalar::ONE` round-trip via to_repr/from_repr.
    //        FAIL → k256 Scalar hits the same shape as Bug A on dalek
    //        Scalar52, widening Bug A's scope.
    //
    // (Originally we also had a slot for `once_cell::sync::Lazy<u64>`
    // first-access — k256 uses `Lazy<[LookupTable; 33]>` for the
    // precomputed generator table — but it required a dev-dep dance
    // for host tests that wasn't worth the noise. If 78/79/80 all PASS
    // and 4/5/74/75 still FAIL, Lazy is implicated by elimination.)
    "k256 encode generator (no mul)",
    "k256 double generator + encode",
    "k256 Scalar::ONE round-trip",
    // Slots 81-83: post-v1.46 re-bisect (Bug A hypothesis falsified by
    // slots 76/77 PASSing; Bug C fixed). New hypotheses for remaining
    // failures:
    //   81 — `u128 >> 52` immediate right shift, the exact shape inside
    //        dalek's montgomery_reduce::part1 and Barrett reduction
    //        generally. LLVM IR expands this into multi-step 64-bit
    //        shifts; if the expansion is wrong, every multi-limb modular
    //        reduction silently produces garbage.
    //   82 — `&'static` data with 4-deep newtype nesting matching k256's
    //        `Scalar(U256(Uint { limbs: [Limb(u64); 4] }))` layout. Slot
    //        77 covered depth-2 (`Wrap([u64; 5])`); this tests whether
    //        deeper GEP through nested newtypes is broken.
    //   83 — `(0..N).rev()` reverse range iterator writing to a stack
    //        array. The one remaining shape in base58_encode_32's
    //        digit-extraction loop that isn't isolated by slots 60-69.
    "arith u128 immediate shr 52",
    "static depth-4 newtype nesting",
    "reverse range iterator write",
    // Slots 84-87: ladder bisect of slot 71's call chain.
    // `Scalar::from_bytes_mod_order(x).to_bytes()` for x=1 expands to:
    //   A: Scalar52::from_bytes(&bytes)         — bit-pack bytes→limbs
    //   B: Scalar52::mul_internal(x, R)         — 5×5 widening mul matrix
    //   C: Scalar52::montgomery_reduce(xR)      — u128 carry chain + L reads
    //   D: result.as_bytes()                     — bit-pack limbs→bytes
    // mul_internal and montgomery_reduce are pub(crate), so we test them
    // via the public `as_montgomery` (B+C together with RR) and
    // `from_montgomery` (just C, synthetic input).
    //   84 — Rung A (from_bytes)
    //   85 — Rung C (from_montgomery on hardcoded R → should yield 1)
    //   86 — Rung B+C (as_montgomery on 1 → should yield R)
    //   87 — Rung D (as_bytes on 1)
    // 84+85+87 PASS, 86 FAIL → mul_internal is broken.
    // 84+86+87 PASS, 85 FAIL → montgomery_reduce broken in isolation.
    // 84+87 PASS, 85+86 FAIL → both broken (or shared subroutine).
    "dalek Scalar52::from_bytes(1)",
    "dalek Scalar52::from_montgomery(R) == 1",
    "dalek Scalar52::ONE.as_montgomery() == R",
    "dalek Scalar52([1,..]).as_bytes() == [1,0,..]",
    // Slots 88-90: post-ladder-PASS investigation. Slot 86 PASSed without
    // the final `Scalar52::sub(result, L)` call that dalek's
    // montgomery_reduce ends with. Now that's restored. If 89 (the
    // re-run of 86 with sub) FAILs while 86 still PASSes (different
    // identity), the bug is specifically in `Scalar52::sub`'s borrow
    // chain or the volatile-load `black_box` — neither of which is
    // covered by any existing tier-1/tier-2 arith slot.
    //   88 — `Scalar52::sub(R, R) == ZERO`: borrow chain, no underflow path.
    //   89 — `Scalar52::sub(ZERO, ONE)`: triggers underflow + conditional-add-L.
    //   90 — full montgomery_reduce-with-sub of widened R → expect ONE.
    "dalek Scalar52::sub(R, R) == 0 (no underflow)",
    "dalek Scalar52::sub(0, 1) underflow path",
    "dalek montgomery_reduce(R) with final sub",
    // Slots 91-93: post-round-2 probes. Ladder rungs 84-90 all PASSed,
    // yet slot 71 still FAILs. The smoking gun: my Scalar52 port uses
    // `a.0[i]` direct field access, but dalek's real Scalar52 implements
    // custom Index/IndexMut traits and uses `a[i]` syntax throughout
    // mul_internal / montgomery_reduce / sub. If `Index<usize>` trait
    // dispatch on a tuple struct miscompiles, my port works but dalek
    // doesn't — exactly the observed pattern.
    //   91 — focused Index/IndexMut trait dispatch probe (custom struct)
    //   92 — `Scalar::ONE.to_bytes()` cross-crate const baseline (no math)
    //   93 — `AffinePoint::GENERATOR.to_encoded_point()` k256 minimal
    //        (no scalar mult, no projective→affine, just const + encode)
    "Index/IndexMut trait dispatch",
    "dalek Scalar::ONE.to_bytes() direct",
    "k256 AffinePoint::GENERATOR.to_encoded_point()",
    // Slots 94-96: Bug F bisect. Slot 91 confirmed the Index-trait bug
    // explaining the dalek-side failures. Slot 93 FAILed showing a
    // *separate* k256 bug in AffinePoint→encoded chain. k256 / elliptic-
    // curve / crypto-bigint have NO Index trait impls (grepped), so 93's
    // bug must be elsewhere in the chain:
    //   subtle::Choice ↔ bool, ConditionallySelectable, or
    //   EncodedPoint::from_affine_coordinates.
    //   94 — `Choice::from(0).into() == false; Choice::from(1).into() == true`.
    //        Tests `From<u8>` and `Into<bool>` on the subtle::Choice newtype.
    //   95 — `u64::conditional_select(&a, &b, Choice(0)) == a; …(Choice(1)) == b`.
    //        Tests `ConditionallySelectable` impl for primitive u64.
    //   96 — `EncodedPoint::from_affine_coordinates(&GX, &GY, true) == G`.
    //        Tests EncodedPoint construction from known good bytes,
    //        bypassing `is_identity()`/`conditional_select` entirely.
    "subtle Choice from(u8) into bool",
    "subtle u64::conditional_select(0|1)",
    "k256 EncodedPoint::from_affine_coordinates(GX, GY)",
    // Slots 97-99: post-round-4 probes.
    //   91 (PREVIOUSLY FAILed, now PASSes!) with black_box runtime index.
    //      But slot 71 still FAILs. Dalek uses CONST literal indices
    //      (s[0], s[1], ...). Need to probe const-index Index dispatch.
    //   96 FAILs but 94/95 PASS — bug is inside `from_affine_coordinates`,
    //      not in Choice or conditional_select. Suspects: GenericArray
    //      Deref impl (uses unsafe ptr cast), or copy_from_slice into it.
    //
    //   97 — `IdxProbe` written/read with LITERAL const indices (mirror
    //        of dalek's `s[0] = …; s[1] = …` pattern). If 97 FAILs, the
    //        bug is specifically constant-index Index trait dispatch.
    //   98 — `GenericArray<u8, U33>::default()` then `ga[i] = v; ga[i]`.
    //        Tests GenericArray's Deref-based indexing. If FAILs, every
    //        GenericArray op is broken → explains slot 96 directly.
    //   99 — `GenericArray<u8, U33>` constructed via `copy_from_slice`.
    //        Tests slice copy into a GenericArray, isolating the exact
    //        operation `from_affine_coordinates` uses.
    "Index trait const-idx (5 writes/reads)",
    "GenericArray<u8, U33> basic index",
    "GenericArray<u8, U33> copy_from_slice",
    // Slots 100-102: post-round-5 probes. Round-5 result: every isolated
    // shape inside Bug-71 and Bug-96 has passed; the bugs only manifest
    // through the actual dalek/sec1 crate compilation. New probes try
    // to narrow further:
    //   100 — local re-impl of sec1's `from_affine_coordinates` body
    //         using raw `[u8; 33]` (no GenericArray). If PASS, sec1's
    //         GenericArray-typed parameter handling is what breaks; if
    //         FAIL, the algorithm shape itself triggers it.
    //   101 — `Tag::compress_y(y.as_slice())` shape: pass `&GenericArray
    //         <u8, U32>` to a function, take `.as_slice().last()` inside.
    //         Tests the only GA-related path slots 98/99 didn't cover.
    //   102 — dalek `Scalar::from_bytes_mod_order([0u8; 32]).to_bytes()`.
    //         All-zero input variant of slot 71. If PASS but 71 FAILs,
    //         the bug is input-dependent (only triggers for non-zero
    //         scalars); if FAIL, the bug is general.
    "k256 from_affine_coords replica (raw [u8; 33])",
    "GenericArray y.as_slice().last() shape",
    "dalek scalar round-trip ZERO",
    // Slots 103-105: one fresh probe per open bug. Most isolated shapes
    // already PASS; these target less-explored surface area.
    //   103 (Bug-71) — `Scalar::from_bytes_mod_order_wide(&[0; 64])`.
    //        Different entry point than slot 71/102 — uses
    //        `from_bytes_wide` → montgomery_mul(R) / montgomery_mul(RR)
    //        composition instead of plain `reduce`. If 71 FAIL but
    //        103 PASS, the bug is in the specific call sequence inside
    //        `Scalar::reduce`, not the wider field of scalar arithmetic.
    //   104 (Bug-96) — `(&[u8; 32]).into() → &FieldBytes<Secp256k1>` then
    //        index. Tests the From-impl conversion that slot 96 uses to
    //        wrap raw byte arrays as `&GenericArray<u8, U32>`. Slot 98
    //        used `GenericArray::default()` to construct; this tests
    //        the conversion-from-raw-array path.
    //   105 (Bug-41) — `base58_encode_32([0; 31] ++ [0x01])`. Single
    //        non-zero byte; expected output is 31 '1's + '2' (= 32 chars).
    //        Tests the digit-extraction loop with `limb_count == 1`
    //        (slot 41 hits much higher limb_counts). If 105 PASS but 41
    //        FAIL, bug requires multi-iter digit extraction.
    "dalek from_bytes_mod_order_wide zero",
    "k256 (&[u8; 32]).into() &FieldBytes",
    "base58 single-nonzero-byte (limb_count=1)",
    // Slots 106-107: post-round-6 breakthrough probes.
    //   106 (Bug-71) — Named-field `struct WrapNamed { bytes: [u8; 32] }`
    //        return-by-value. dalek's `Scalar` is exactly this shape.
    //        Slot 70 (return [u8; 32] direct) and slot 84 (return tuple-
    //        struct Scalar52 with pub field) both PASS. Slot 71/102/103
    //        (return Scalar with pub(crate) field) all FAIL. Suspect:
    //        named-field-struct-wrapping-array return ABI is broken.
    //   107 (Bug-41) — Hand-rolled base58 of [0;31]+[0x01], without the
    //        `seq!` macro (which unrolls 8 iterations of the outer loop
    //        in `base58_encode_32`). Plain Rust loops. If 107 PASS but
    //        105 FAIL, the bug is in the `seq!` expansion specifically.
    "named-field struct return (Scalar shape)",
    "base58 hand-rolled no-seq! single-nonzero",
    // Slots 108-109: post-round-7 probes. Slots 106/107 PASS falsified
    // struct-return-ABI (Bug-71) and seq!-macro (Bug-41) hypotheses.
    //   108 (Bug-41) — `<[u8]>::reverse()` on a partial sub-slice. The
    //        only operation in `base58_encode_32` that slot 107 hand-rolls
    //        differently (slot 107 uses manual swap; original uses
    //        `output[..result_len].reverse()`). If 108 FAIL → that's the
    //        Bug-41 repro.
    //   109 (Bug-71) — `Scalar::from_bytes_mod_order([0; 32]) == Scalar::ZERO`
    //        via dalek's `PartialEq` instead of comparing bytes. Separates
    //        "is the Scalar value correct?" from "is to_bytes broken?".
    //        If 109 PASS but 102 FAIL → bug is specifically in `to_bytes`.
    //        If 109 FAIL → the Scalar value itself is wrong.
    "<[u8]>::reverse() partial sub-slice",
    "dalek Scalar(0) == Scalar::ZERO (no to_bytes)",
    // Slots 110-112: post-round-8 probes. Slot 108 confirmed Bug-41
    // minimal repro (slice.reverse partial sub-slice). Slot 109 FAIL
    // confirms Bug-71 is value-level (not just to_bytes). Slot 101
    // flipped PASS — our previous Bug-96 minimal repro no longer
    // reproduces, so we need a new shape.
    //   110 (Bug-96) — `dst_ga.copy_from_slice(src_ga_as_slice)`. Slot 99
    //        used `&[u8; 32]` source; this uses `&GenericArray` source,
    //        the actual shape used by `from_affine_coordinates(x, y, _)`.
    //   111 (Bug-71) — `Scalar::ZERO == Scalar::ZERO`. Pure const-vs-
    //        const PartialEq. If FAIL, PartialEq is broken; if PASS,
    //        slot 109's FAIL is genuinely from from_bytes_mod_order
    //        returning wrong value.
    //   112 (Bug-71) — `Scalar::from_canonical_bytes([0;32]).unwrap() ==
    //        Scalar::ZERO`. Alternate entry point that doesn't go through
    //        `reduce()`. If 112 PASS but 109 FAIL, the bug is in
    //        `Scalar::reduce()` specifically.
    "GenericArray dst.copy_from_slice(src GA)",
    "dalek Scalar::ZERO == Scalar::ZERO (PartialEq)",
    "dalek Scalar::from_canonical_bytes(0) == ZERO",
    // Slots 113-115: post-round-9 narrowing. Slot 112 FAIL combined with
    // slot 111 PASS pins Bug-71 to `Scalar::reduce()` on zero input
    // (since `from_canonical_bytes(0)` only goes through `from_bits` +
    // `is_canonical()` which itself calls `reduce()`, and PartialEq
    // itself works). The verbatim Scalar52 port (slots 84-90) covered
    // each reduce step for value=1/R but NEVER tested zero input.
    //   113 (Bug-71) — `bisect::Scalar52::from_bytes(&[0; 32]).0 == [0; 5]`.
    //        Zero unpack via the verbatim port.
    //   114 (Bug-71) — `bisect::Scalar52::mul_internal(&ZERO, &R) == [0; 9]`.
    //        Zero · R widening multiply via the verbatim port.
    //   115 (Bug-71) — `bisect::Scalar52::montgomery_reduce(&[0; 9]).0 == [0; 5]`.
    //        Full reduce of zero-widened limbs via the verbatim port.
    // If all three PASS, Bug-71 is real-dalek-codegen-vs-port (same source,
    // different compilation). If any FAIL, that step is the smoking gun
    // and we have our minimal repro.
    "dalek bisect Scalar52::from_bytes(0)",
    "dalek bisect Scalar52::mul_internal(0, R)",
    "dalek bisect Scalar52::montgomery_reduce(0)",
    // Slots 116-117: post-round-10 probes. Round-10 win: Bug-41 FIXED
    // (slot 108 + cascade all PASS). Slots 113-115 ALL PASS = every
    // individual step of `Scalar::reduce()` works on zero in our verbatim
    // port. But real dalek `reduce()` on zero still FAILs (slot 102/109/112).
    //   116 (Bug-71) — `Scalar52::ZERO.as_bytes() == [0; 32]`. Closes the
    //        every-primitive-on-zero loop (slot 87 was pack of ONE; this
    //        is pack of ZERO).
    //   117 (Bug-71) — Compose the full `reduce()` pipeline inside our
    //        crate on zero input: from_bytes → mul_internal(R) →
    //        montgomery_reduce → as_bytes. If FAIL, we've reproduced
    //        Bug-71 in `logic` (huge — minimal repro at last). If PASS,
    //        Bug-71 is a cross-crate-composition bug (real dalek's
    //        reduce only fires the bug when monomorphized in dalek's
    //        crate, not ours).
    "dalek bisect Scalar52::ZERO.as_bytes()",
    "dalek bisect full reduce pipeline (zero in)",
    // Slots 118–152: P-256 and RSA primitives, stages, and boundaries.
    "p256 public key hmac derivation",
    "p256 public key scalar derivation",
    "p256 public key generator",
    "p256 public key point double",
    "p256 public key zero scalar rejected",
    "p256 public key order scalar rejected",
    "p256 public key x encoding",
    "p256 public key y encoding",
    "p256 signature rfc6979 sample",
    "p256 signature rfc6979 test",
    "p256 signature ephemeral r",
    "p256 signature ephemeral signature",
    "p256 signature zero nonce rejected",
    "p256 signature low s",
    "p256 signature high s",
    "p256 signature message window carry",
    "p256 signature ephemeral hmac",
    "rsa pss sha256",
    "rsa pss mgf1 partial block",
    "rsa pss salt32 encoding",
    "rsa pss empty salt encoding",
    "rsa pss maximum salt encoding",
    "rsa pss oversized salt rejected",
    "rsa pss salt carry",
    "rsa pss crt known answer",
    "rsa pss crt fault rejected",
    "rsa pss crt modulus rejected",
    "rsa modulus multiplication carry",
    "rsa modulus progression carry",
    "rsa modulus prime filter",
    "rsa modulus pseudoprime rejected",
    "rsa modulus zero stride rejected",
    "rsa modulus upper bound rejected",
    "rsa modulus equal factors rejected",
    "rsa modulus undersized factor rejected",
    "end-to-end p256 public candidate pipeline",
    "end-to-end p256 signature candidate pipeline",
    "end-to-end rsa pss candidate pipeline",
    "end-to-end rsa modulus candidate pipeline",
];

pub fn run_self_test(results: &mut [u32]) {
    #[cfg(feature = "self_test_solana")]
    {
        results[0] = check_primitive_xoroshiro();
        results[1] = check_primitive_sha512();
        results[2] = check_primitive_ed25519();
        results[3] = check_primitive_base58();
        results[10] = check_solana_priv();
        results[11] = check_solana_pub();
        results[12] = check_solana_encoded();
        results[31] = check_arith_u32_div_var();
        results[32] = check_arith_u32_div_const();
        results[33] = check_arith_u64_div_var();
        results[34] = check_arith_u64_div_const();
        results[35] = check_arith_u32_rem_var();
        results[36] = check_arith_u64_rem_var();
        results[37] = check_arith_u32_mul_lo();
        results[38] = check_arith_u64_mul_lo();
        results[39] = check_arith_u64_mul_hi();
        results[40] = check_arith_u128_mul();
        results[41] = check_base58_var_len();
        results[43] = check_base58_all_zeros();
        results[46] = check_arith_overflowing_add();
        results[47] = check_arith_overflowing_sub();
        results[48] = check_arith_carry_chain_3limb();
        results[49] = check_arith_widening_mul_pair();
        results[50] = check_arith_mad_lo_u64();
        results[51] = check_arith_mad_hi_u64();
        results[52] = check_arith_mul_wide_u32();
        results[53] = check_arith_mask_blend_true();
        results[54] = check_arith_mask_blend_false();
        results[55] = check_arith_var_shr_u64();
        results[56] = check_arith_var_shl_u64();
        results[57] = check_arith_blackbox_identity_u64();
        results[58] = check_arith_blackbox_identity_u32();
        results[59] = check_base58_div_by_58();
        results[60] = check_iter_static_table_lookup();
        results[61] = check_iter_mut_slice_partial();
        results[62] = check_iter_mut_alphabet_lookup();
        results[63] = check_iter_static_slice_lookup();
        results[64] = check_arith_divrem_by_58_pow_5();
        results[65] = check_arith_i128_chain_add();
        results[66] = check_base58_limb_divrem();
        results[67] = check_dynamic_index_write();
        results[68] = check_arith_widening_mul_chain_3term();
        results[69] = check_base58_inner_mutate_phase();
        results[70] = check_dalek_clamp_integer();
        results[71] = check_dalek_scalar_round_trip_one();
        results[72] = check_dalek_mul_base_scalar_one();
        results[81] = check_arith_u128_imm_shr_52();
        results[82] = check_static_depth4_newtype_nesting();
        results[83] = check_reverse_range_write();
        results[84] = check_dalek_scalar52_from_bytes();
        results[85] = check_dalek_scalar52_montgomery_reduce_r();
        results[86] = check_dalek_scalar52_mul_internal_then_reduce_one_r();
        results[87] = check_dalek_scalar52_as_bytes_one();
        results[88] = check_dalek_scalar52_sub_no_underflow();
        results[89] = check_dalek_scalar52_sub_with_underflow();
        results[90] = check_dalek_scalar52_montgomery_reduce_with_sub();
        results[91] = check_index_trait_dispatch();
        results[92] = check_dalek_scalar_one_to_bytes_direct();
        results[102] = check_dalek_scalar_round_trip_zero();
        results[103] = check_dalek_scalar_from_bytes_wide_zero();
        results[105] = check_base58_min_nonzero();
        results[106] = check_named_field_struct_return();
        results[107] = check_base58_handrolled_no_seq();
        results[108] = check_slice_reverse_partial();
        results[109] = check_dalek_scalar_eq_zero();
        results[111] = check_dalek_zero_eq_zero();
        results[112] = check_dalek_from_canonical_zero();
        results[113] = check_dalek_scalar52_from_bytes_zero();
        results[114] = check_dalek_scalar52_mul_internal_zero();
        results[115] = check_dalek_scalar52_montgomery_reduce_zero();
        results[116] = check_dalek_scalar52_as_bytes_zero();
        results[117] = check_dalek_reduce_pipeline_zero();
    }
    #[cfg(feature = "self_test_bitcoin")]
    {
        results[4] = check_primitive_secp256k1_compressed();
        results[7] = check_primitive_ripemd160();
        results[8] = check_primitive_sha256_32();
        results[16] = check_bitcoin_priv();
        results[17] = check_bitcoin_pub();
        results[18] = check_bitcoin_pkh();
        results[19] = check_bitcoin_encoded();
        results[20] = check_bitcoin_matches();
        results[21] = check_wif_compressed_mainnet();
        results[22] = check_wif_uncompressed_mainnet();
        results[23] = check_wif_compressed_testnet();
        results[24] = check_wif_uncompressed_testnet();
        results[42] = check_base58_var_len_leading_zero();
        results[45] = check_bech32_p2wpkh();
        results[73] = check_k256_secret_from_bytes_one();
        results[74] = check_k256_derive_scalar_one();
        results[75] = check_k256_derive_scalar_two();
        results[76] = check_static_u64_array_lookup();
        results[77] = check_static_struct_wrapped_u64_lookup();
        results[78] = check_k256_encode_generator();
        results[79] = check_k256_double_generator();
        results[80] = check_k256_scalar_one_round_trip();
        results[93] = check_k256_affine_generator_encode();
        results[94] = check_subtle_choice_u8_into_bool();
        results[95] = check_subtle_conditional_select_u64();
        results[96] = check_k256_encoded_point_from_affine_coords();
        results[97] = check_index_trait_const_indices();
        results[98] = check_generic_array_basic_index();
        results[99] = check_generic_array_copy_from_slice();
        results[100] = check_from_affine_coords_replica();
        results[101] = check_generic_array_as_slice_last();
        results[104] = check_field_bytes_into_conversion();
        results[110] = check_generic_array_copy_from_ga_source();
    }
    #[cfg(feature = "self_test_ethereum")]
    {
        results[5] = check_primitive_secp256k1_uncompressed();
        results[6] = check_primitive_keccak256();
        results[13] = check_ethereum_priv();
        results[14] = check_ethereum_pub();
        results[15] = check_ethereum_address();
    }
    #[cfg(feature = "self_test_shallenge")]
    {
        results[9] = check_primitive_sha256_variable();
        results[25] = check_shallenge_hash();
        results[26] = check_shallenge_nonce_len();
        results[27] = check_shallenge_is_better();
        results[28] = check_compare_hashes_lt();
        results[29] = check_compare_hashes_gt();
        results[30] = check_compare_hashes_eq();
        results[44] = check_xoroshiro_base64_nonce();
    }
    #[cfg(feature = "self_test_p256_public_key")]
    {
        results[118] = check_p256_public_key_hmac_derivation();
        results[119] = check_p256_public_key_scalar_derivation();
        results[120] = check_p256_public_key_generator();
        results[121] = check_p256_public_key_point_double();
        results[122] = check_p256_public_key_zero_scalar_rejected();
        results[123] = check_p256_public_key_order_scalar_rejected();
        results[124] = check_p256_public_key_x_encoding();
        results[125] = check_p256_public_key_y_encoding();
        results[153] = check_p256_public_end_to_end();
    }
    #[cfg(feature = "self_test_p256_signature")]
    {
        results[126] = check_p256_signature_rfc6979_sample();
        results[127] = check_p256_signature_rfc6979_test();
        results[128] = check_p256_signature_ephemeral_r();
        results[129] = check_p256_signature_ephemeral_signature();
        results[130] = check_p256_signature_zero_nonce_rejected();
        results[131] = check_p256_signature_low_s();
        results[132] = check_p256_signature_high_s();
        results[133] = check_p256_signature_message_window_carry();
        results[134] = check_p256_signature_ephemeral_hmac();
        results[154] = check_p256_signature_end_to_end();
    }
    #[cfg(feature = "self_test_rsa_pss")]
    {
        results[135] = check_rsa_pss_sha256();
        results[136] = check_rsa_pss_mgf1_partial_block();
        results[137] = check_rsa_pss_salt32_encoding();
        results[138] = check_rsa_pss_empty_salt_encoding();
        results[139] = check_rsa_pss_maximum_salt_encoding();
        results[140] = check_rsa_pss_oversized_salt_rejected();
        results[141] = check_rsa_pss_salt_carry();
        results[142] = check_rsa_pss_crt_known_answer();
        results[143] = check_rsa_pss_crt_fault_rejected();
        results[144] = check_rsa_pss_crt_modulus_rejected();
        results[155] = check_rsa_pss_end_to_end();
    }
    #[cfg(feature = "self_test_rsa_modulus")]
    {
        results[145] = check_rsa_modulus_multiplication_carry();
        results[146] = check_rsa_modulus_progression_carry();
        results[147] = check_rsa_modulus_prime_filter();
        results[148] = check_rsa_modulus_pseudoprime_rejected();
        results[149] = check_rsa_modulus_zero_stride_rejected();
        results[150] = check_rsa_modulus_upper_bound_rejected();
        results[151] = check_rsa_modulus_equal_factors_rejected();
        results[152] = check_rsa_modulus_undersized_factor_rejected();
        results[156] = check_rsa_modulus_end_to_end();
    }
}

#[cfg(all(test, feature = "self_test"))]
mod test {
    use super::*;

    #[test]
    fn all_self_test_checks_pass_on_cpu() {
        let mut results = [0u32; SELF_TEST_NUM_CHECKS];
        run_self_test(&mut results);
        for (i, &r) in results.iter().enumerate() {
            assert_eq!(
                r, 1,
                "self-test check {} ({}) failed",
                i, SELF_TEST_LABELS[i]
            );
        }
    }
}
