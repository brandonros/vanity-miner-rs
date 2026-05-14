//! On-device / on-CPU self-test: runs known-answer tests for every logic
//! primitive against externally-validated expected values, writing
//! pass(1)/fail(0) per check into the results buffer.
//!
//! Each slot has a dedicated `check_*` function. GPU mode launches one
//! kernel per slot so an illegal-address fault localizes to a single
//! subsystem (and the kernels before it still produce reliable results
//! before the context goes sticky-errored). CPU mode's `run_self_test`
//! calls them all in sequence.

use crate::{
    BitcoinVanityKeyRequest, BitcoinVanityKeyResult, EthereumVanityKeyRequest,
    EthereumVanityKeyResult, ShallengeRequest, ShallengeResult, SolanaVanityKeyRequest,
    SolanaVanityKeyResult, base58_encode, base58_encode_32, compare_hashes,
    ed25519_derive_public_key, encode_p2wpkh_address, generate_and_check_bitcoin_vanity_key,
    generate_and_check_ethereum_vanity_key, generate_and_check_shallenge,
    generate_and_check_solana_vanity_key, generate_base64_nonce, generate_random_private_key,
    keccak256_64bytes, private_key_to_wif, ripemd160_32bytes_from_bytes,
    secp256k1_derive_public_key, secp256k1_derive_public_key_uncompressed,
    sha256_32_from_bytes, sha256_from_bytes, sha512_32bytes_from_bytes,
};

pub const SELF_TEST_NUM_CHECKS: usize = 113;

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
];

// === Solana per-primitive bisect (slots 0-3) ===
// The `solana priv` slot ran the *whole* pipeline before checking the priv
// bytes; if that kernel faulted we couldn't tell which primitive triggered
// it. These four `check_primitive_*` functions exercise each stage in
// isolation against externally-validated intermediates, so GPU mode can
// localize a fault to xoroshiro / sha512 / ed25519 / base58.

const SOLANA_PRIMITIVE_PRIV: [u8; 32] = [
    0xfa, 0x9c, 0xe9, 0xb0, 0x2d, 0xc2, 0x8a, 0x48,
    0xf7, 0xe9, 0xd1, 0x55, 0x06, 0xd3, 0xd2, 0xc4,
    0x43, 0xd5, 0x96, 0x56, 0x5f, 0xa0, 0x52, 0x14,
    0xb0, 0xff, 0x7c, 0x5a, 0xb5, 0xe7, 0x95, 0x6b,
];

const SOLANA_PRIMITIVE_HASHED_PRIV: [u8; 64] = [
    0xaa, 0xe4, 0x1d, 0x15, 0x43, 0x8a, 0x30, 0xa5,
    0x0e, 0x27, 0x4b, 0x13, 0x6d, 0x5c, 0x2a, 0x7c,
    0x36, 0x6e, 0x68, 0xbf, 0xf9, 0xa0, 0xbb, 0x05,
    0x87, 0x2c, 0x35, 0x75, 0x2e, 0x9a, 0x45, 0xa4,
    0x8c, 0x25, 0x5f, 0x21, 0xb8, 0x43, 0xfc, 0xa7,
    0x21, 0x81, 0x3f, 0xc2, 0x40, 0x3e, 0x20, 0x13,
    0xe0, 0xe8, 0x1d, 0xd6, 0xd7, 0xc9, 0xd8, 0x69,
    0xac, 0xf6, 0x03, 0x1e, 0x33, 0xb6, 0x95, 0x6a,
];

const SOLANA_PRIMITIVE_PUB: [u8; 32] = [
    0x08, 0x9a, 0x23, 0xff, 0xc4, 0x22, 0xf5, 0x3d,
    0x11, 0x45, 0x87, 0x01, 0x2b, 0xb2, 0xc0, 0x28,
    0x49, 0x2f, 0xab, 0xda, 0xbe, 0x12, 0x66, 0xbc,
    0x9a, 0xd6, 0x69, 0x8a, 0xc4, 0x30, 0x16, 0xbb,
];

pub fn check_primitive_xoroshiro() -> u32 {
    let priv_key = generate_random_private_key(3, 583437459223573146);
    (priv_key == SOLANA_PRIMITIVE_PRIV) as u32
}

pub fn check_primitive_sha512() -> u32 {
    let hashed = sha512_32bytes_from_bytes(&SOLANA_PRIMITIVE_PRIV);
    (hashed == SOLANA_PRIMITIVE_HASHED_PRIV) as u32
}

pub fn check_primitive_ed25519() -> u32 {
    let pub_key = ed25519_derive_public_key(&SOLANA_PRIMITIVE_HASHED_PRIV);
    (pub_key == SOLANA_PRIMITIVE_PUB) as u32
}

pub fn check_primitive_base58() -> u32 {
    let expected: &[u8] = b"aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC";
    let mut out = [0u8; 64];
    let n = base58_encode_32(&SOLANA_PRIMITIVE_PUB, &mut out);
    (n == expected.len() && bytes_eq_prefix(&out, expected)) as u32
}

// === Non-solana primitive bisect (slots 4-9) ===
// Same idea as slots 0-3, but for the primitives consumed by the bitcoin /
// ethereum / shallenge / WIF pipelines. Each KAT pair is taken from the
// per-module unit tests in the corresponding `logic/src/*.rs` file, so a
// fault here means the primitive itself is broken on the device — separate
// from a fault in a composed pipeline kernel that just inlines it.

const SECP256K1_PRIMITIVE_PRIV: [u8; 32] = [
    0x15, 0x2d, 0x53, 0x72, 0x3d, 0xa4, 0x20, 0x34,
    0x78, 0x57, 0x4b, 0x15, 0x31, 0x43, 0xa7, 0xea,
    0xa9, 0x21, 0xa8, 0xd8, 0x2c, 0x62, 0x95, 0x17,
    0xd6, 0xb1, 0x89, 0x49, 0xf0, 0x11, 0x1a, 0xbb,
];

const SECP256K1_PRIMITIVE_COMPRESSED_PUB: [u8; 33] = [
    0x03, 0x91, 0x63, 0xab, 0x44, 0x9d, 0x4b, 0x90,
    0xde, 0x13, 0xce, 0x60, 0xb5, 0x04, 0xbf, 0xc2,
    0x7a, 0x4a, 0xed, 0x37, 0x8c, 0x1f, 0x83, 0x38,
    0x68, 0x61, 0x56, 0xb9, 0x14, 0x45, 0x63, 0x7c,
    0x8d,
];

const SECP256K1_PRIMITIVE_UNCOMPRESSED_PUB: [u8; 65] = [
    0x04, 0x91, 0x63, 0xab, 0x44, 0x9d, 0x4b, 0x90,
    0xde, 0x13, 0xce, 0x60, 0xb5, 0x04, 0xbf, 0xc2,
    0x7a, 0x4a, 0xed, 0x37, 0x8c, 0x1f, 0x83, 0x38,
    0x68, 0x61, 0x56, 0xb9, 0x14, 0x45, 0x63, 0x7c,
    0x8d, 0x33, 0x27, 0x2b, 0x79, 0x99, 0x4d, 0xae,
    0x54, 0xda, 0x40, 0x11, 0xcc, 0x3e, 0x34, 0x91,
    0xcc, 0xdf, 0x3b, 0xd3, 0xfd, 0x92, 0x97, 0x8a,
    0x00, 0x87, 0x37, 0x27, 0xf9, 0x9b, 0xeb, 0x43,
    0x75,
];

const KECCAK256_PRIMITIVE_INPUT: [u8; 64] = [
    0x61, 0xa3, 0x14, 0xb0, 0x18, 0x37, 0x24, 0xea,
    0x0e, 0x5f, 0x23, 0x75, 0x84, 0xcb, 0x76, 0x09,
    0x2e, 0x25, 0x3b, 0x99, 0x78, 0x3d, 0x84, 0x6a,
    0x5b, 0x10, 0xdb, 0x15, 0x51, 0x28, 0xea, 0xfd,
    0x61, 0xa3, 0x14, 0xb0, 0x18, 0x37, 0x24, 0xea,
    0x0e, 0x5f, 0x23, 0x75, 0x84, 0xcb, 0x76, 0x09,
    0x2e, 0x25, 0x3b, 0x99, 0x78, 0x3d, 0x84, 0x6a,
    0x5b, 0x10, 0xdb, 0x15, 0x51, 0x28, 0xea, 0xfd,
];

const KECCAK256_PRIMITIVE_OUTPUT: [u8; 32] = [
    0x0f, 0x43, 0x9a, 0x98, 0x30, 0x55, 0x8b, 0x9c,
    0xd6, 0x84, 0x23, 0x28, 0xdd, 0x11, 0x58, 0x54,
    0x01, 0xc3, 0x43, 0x21, 0xa5, 0x3b, 0x29, 0x42,
    0x2a, 0xac, 0xde, 0x31, 0x06, 0x43, 0xd3, 0x73,
];

// "brandonros/000000000000000000000" — 32 ASCII bytes.
const HASH_PRIMITIVE_INPUT_32: [u8; 32] = *b"brandonros/000000000000000000000";

const RIPEMD160_PRIMITIVE_OUTPUT: [u8; 20] = [
    0xce, 0xf7, 0x32, 0xce, 0xe6, 0x7e, 0xa5, 0xd8,
    0x1d, 0x08, 0x70, 0x8b, 0x22, 0xbf, 0x1f, 0xc7,
    0x91, 0x1d, 0x32, 0x09,
];

const SHA256_PRIMITIVE_OUTPUT_32: [u8; 32] = [
    0xf7, 0xa4, 0x1d, 0xae, 0x11, 0x96, 0x28, 0x2f,
    0x0a, 0x54, 0x4a, 0x8c, 0x7f, 0x1b, 0xbf, 0x61,
    0xbd, 0xa7, 0x93, 0x07, 0xdc, 0x42, 0x4c, 0x0d,
    0x9f, 0xeb, 0xd2, 0x7b, 0x08, 0xe1, 0xbf, 0x78,
];

// 33 ASCII bytes — forces the variable-length sha256 path through one
// complete 64-byte block plus padding.
const HASH_PRIMITIVE_INPUT_33: [u8; 33] = *b"brandonros/0000000000000000000000";

const SHA256_PRIMITIVE_OUTPUT_VARIABLE: [u8; 32] = [
    0x06, 0x23, 0x89, 0x93, 0x6c, 0x51, 0x9e, 0xd7,
    0x3f, 0x33, 0x71, 0xef, 0x2e, 0x66, 0xd4, 0x38,
    0xe1, 0xcf, 0x0a, 0x66, 0x03, 0xf8, 0xb6, 0x7c,
    0x74, 0x8a, 0x5d, 0x21, 0x1e, 0x48, 0xb2, 0x9d,
];

pub fn check_primitive_secp256k1_compressed() -> u32 {
    let pub_key = secp256k1_derive_public_key(&SECP256K1_PRIMITIVE_PRIV);
    (pub_key == SECP256K1_PRIMITIVE_COMPRESSED_PUB) as u32
}

pub fn check_primitive_secp256k1_uncompressed() -> u32 {
    let pub_key = secp256k1_derive_public_key_uncompressed(&SECP256K1_PRIMITIVE_PRIV);
    (pub_key == SECP256K1_PRIMITIVE_UNCOMPRESSED_PUB) as u32
}

pub fn check_primitive_keccak256() -> u32 {
    let hash = keccak256_64bytes(&KECCAK256_PRIMITIVE_INPUT);
    (hash == KECCAK256_PRIMITIVE_OUTPUT) as u32
}

pub fn check_primitive_ripemd160() -> u32 {
    let hash = ripemd160_32bytes_from_bytes(&HASH_PRIMITIVE_INPUT_32);
    (hash == RIPEMD160_PRIMITIVE_OUTPUT) as u32
}

pub fn check_primitive_sha256_32() -> u32 {
    let hash = sha256_32_from_bytes(&HASH_PRIMITIVE_INPUT_32);
    (hash == SHA256_PRIMITIVE_OUTPUT_32) as u32
}

pub fn check_primitive_sha256_variable() -> u32 {
    let hash = sha256_from_bytes(&HASH_PRIMITIVE_INPUT_33);
    (hash == SHA256_PRIMITIVE_OUTPUT_VARIABLE) as u32
}

/// Compare the first `expected.len()` bytes of `actual` to `expected`.
fn bytes_eq_prefix(actual: &[u8; 64], expected: &[u8]) -> bool {
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

// === Solana (rng_seed=583437459223573146, thread_idx=3) ===

fn solana_test() -> SolanaVanityKeyResult {
    let req = SolanaVanityKeyRequest {
        prefix: b"",
        suffix: b"",
        thread_idx: 3,
        rng_seed: 583437459223573146,
    };
    generate_and_check_solana_vanity_key(&req)
}

pub fn check_solana_priv() -> u32 {
    let expected: [u8; 32] = [
        0xfa, 0x9c, 0xe9, 0xb0, 0x2d, 0xc2, 0x8a, 0x48,
        0xf7, 0xe9, 0xd1, 0x55, 0x06, 0xd3, 0xd2, 0xc4,
        0x43, 0xd5, 0x96, 0x56, 0x5f, 0xa0, 0x52, 0x14,
        0xb0, 0xff, 0x7c, 0x5a, 0xb5, 0xe7, 0x95, 0x6b,
    ];
    (solana_test().private_key == expected) as u32
}

pub fn check_solana_pub() -> u32 {
    let expected: [u8; 32] = [
        0x08, 0x9a, 0x23, 0xff, 0xc4, 0x22, 0xf5, 0x3d,
        0x11, 0x45, 0x87, 0x01, 0x2b, 0xb2, 0xc0, 0x28,
        0x49, 0x2f, 0xab, 0xda, 0xbe, 0x12, 0x66, 0xbc,
        0x9a, 0xd6, 0x69, 0x8a, 0xc4, 0x30, 0x16, 0xbb,
    ];
    (solana_test().public_key == expected) as u32
}

pub fn check_solana_encoded() -> u32 {
    let expected: &[u8] = b"aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC";
    let sol = solana_test();
    (sol.encoded_len == expected.len()
        && bytes_eq_prefix(&sol.encoded_public_key, expected)) as u32
}

// === Ethereum (rng_seed=15455378110306975741, thread_idx=0) ===

fn ethereum_test() -> EthereumVanityKeyResult {
    let req = EthereumVanityKeyRequest {
        prefix: b"",
        suffix: b"",
        thread_idx: 0,
        rng_seed: 15455378110306975741,
    };
    generate_and_check_ethereum_vanity_key(&req)
}

pub fn check_ethereum_priv() -> u32 {
    let expected: [u8; 32] = [
        0x1c, 0xcf, 0x23, 0x85, 0x14, 0x11, 0x73, 0x04,
        0x8c, 0x0d, 0x06, 0xc1, 0x07, 0x08, 0x69, 0xa1,
        0x6b, 0xf6, 0x3b, 0x69, 0x71, 0x66, 0x33, 0xe9,
        0xbf, 0xe7, 0x9a, 0x98, 0x13, 0xc4, 0x05, 0xab,
    ];
    (ethereum_test().private_key == expected) as u32
}

pub fn check_ethereum_pub() -> u32 {
    let expected: [u8; 64] = [
        0x88, 0xf1, 0xff, 0xe7, 0x4d, 0x7c, 0x83, 0xb6,
        0xae, 0xe0, 0xc7, 0x0f, 0x42, 0x38, 0xf5, 0xaa,
        0x91, 0x7b, 0x80, 0x62, 0xc9, 0xd3, 0x78, 0xfd,
        0xf4, 0x04, 0x2c, 0xcc, 0xdc, 0xca, 0x26, 0x39,
        0x42, 0x4c, 0x5d, 0xb5, 0x21, 0x21, 0x6a, 0xb6,
        0xb7, 0x65, 0xd9, 0xf6, 0x37, 0x8e, 0xe9, 0x26,
        0x11, 0x8a, 0xbf, 0xf8, 0xaf, 0x52, 0x4e, 0x0a,
        0x5d, 0x5e, 0x82, 0x75, 0x28, 0x6d, 0xd4, 0xc9,
    ];
    (ethereum_test().public_key == expected) as u32
}

pub fn check_ethereum_address() -> u32 {
    let expected: [u8; 20] = [
        0x55, 0x55, 0x63, 0x59, 0x0c, 0x72, 0x4a, 0x58,
        0xf7, 0xbb, 0x48, 0xb6, 0xc8, 0x47, 0xaa, 0x63,
        0x1a, 0x48, 0x65, 0x1c,
    ];
    (ethereum_test().address == expected) as u32
}

// === Bitcoin (rng_seed=13278869120712471092, thread_idx=1) ===

fn bitcoin_test() -> BitcoinVanityKeyResult {
    let req = BitcoinVanityKeyRequest {
        prefix: b"bc1q",
        suffix: b"",
        thread_idx: 1,
        rng_seed: 13278869120712471092,
    };
    generate_and_check_bitcoin_vanity_key(&req)
}

pub fn check_bitcoin_priv() -> u32 {
    let expected: [u8; 32] = [
        0x36, 0x32, 0xf6, 0x6f, 0xed, 0x3b, 0x77, 0xf3,
        0x30, 0x9c, 0x86, 0xd7, 0x08, 0xfc, 0xce, 0x8a,
        0x07, 0x1a, 0x61, 0xa1, 0xa9, 0x4a, 0xdd, 0x0c,
        0xb4, 0x5f, 0x95, 0x7c, 0x34, 0x67, 0xd1, 0xdc,
    ];
    (bitcoin_test().private_key == expected) as u32
}

pub fn check_bitcoin_pub() -> u32 {
    let expected: [u8; 33] = [
        0x02, 0x54, 0x38, 0x15, 0x68, 0x27, 0x6c, 0x32,
        0xfe, 0x4a, 0x16, 0x77, 0xbb, 0x97, 0xb2, 0x62,
        0x9f, 0xcf, 0x68, 0x4e, 0x3e, 0x22, 0xcb, 0x4d,
        0x95, 0xfa, 0x1c, 0x53, 0x60, 0xa0, 0xe7, 0x79,
        0xbf,
    ];
    (bitcoin_test().public_key == expected) as u32
}

pub fn check_bitcoin_pkh() -> u32 {
    let expected: [u8; 20] = [
        0x00, 0x01, 0xb5, 0x3d, 0x6d, 0x26, 0xf1, 0x8c,
        0x85, 0xbf, 0xf2, 0xac, 0x3c, 0x57, 0x1e, 0xe7,
        0xe0, 0xc8, 0x87, 0xff,
    ];
    (bitcoin_test().public_key_hash == expected) as u32
}

pub fn check_bitcoin_encoded() -> u32 {
    let expected: &[u8] = b"bc1qqqqm20tdymccepdl72krc4c7ulsv3pllzju9s4";
    let btc = bitcoin_test();
    (btc.encoded_len == expected.len()
        && bytes_eq_prefix(&btc.encoded_public_key, expected)) as u32
}

pub fn check_bitcoin_matches() -> u32 {
    bitcoin_test().matches as u32
}

// === WIF (4 flag combinations) ===
// Standalone — doesn't depend on the bitcoin search; just feeds the known
// private key into private_key_to_wif with each flag combo.

const BITCOIN_TEST_PRIV: [u8; 32] = [
    0x36, 0x32, 0xf6, 0x6f, 0xed, 0x3b, 0x77, 0xf3,
    0x30, 0x9c, 0x86, 0xd7, 0x08, 0xfc, 0xce, 0x8a,
    0x07, 0x1a, 0x61, 0xa1, 0xa9, 0x4a, 0xdd, 0x0c,
    0xb4, 0x5f, 0x95, 0x7c, 0x34, 0x67, 0xd1, 0xdc,
];

pub fn check_wif_compressed_mainnet() -> u32 {
    let mut wif_buf = [0u8; 64];
    let n = private_key_to_wif(&BITCOIN_TEST_PRIV, true, false, &mut wif_buf);
    (n == 52
        && bytes_eq_prefix(&wif_buf, b"Ky34pxSf7FLh6GFgKpvJwfDFdCw6GG4vytEh3Kt3ZzZoxw3e3WaG")) as u32
}

pub fn check_wif_uncompressed_mainnet() -> u32 {
    let mut wif_buf = [0u8; 64];
    let n = private_key_to_wif(&BITCOIN_TEST_PRIV, false, false, &mut wif_buf);
    (n == 51
        && bytes_eq_prefix(&wif_buf, b"5JEA2MGL4EDcpQr6HVywMzbVgvTJWHZA4NaTk7znSbnx3ooTWrv")) as u32
}

pub fn check_wif_compressed_testnet() -> u32 {
    let mut wif_buf = [0u8; 64];
    let n = private_key_to_wif(&BITCOIN_TEST_PRIV, true, true, &mut wif_buf);
    (n == 52
        && bytes_eq_prefix(&wif_buf, b"cPQ4HsSWYK2xFhiwiEjSJyiKFSEVviAd3vPA9kLZ57DpDg5McHdr")) as u32
}

pub fn check_wif_uncompressed_testnet() -> u32 {
    let mut wif_buf = [0u8; 64];
    let n = private_key_to_wif(&BITCOIN_TEST_PRIV, false, true, &mut wif_buf);
    (n == 51
        && bytes_eq_prefix(&wif_buf, b"91znc65seTHknUMNuqsrEb9TLap1fT6MQKSQpkMHnLXzpohhjJo")) as u32
}

// === Shallenge (rng_seed=12345, thread_idx=0, "brandonros", target=max) ===

fn shallenge_test() -> ShallengeResult {
    let user = *b"brandonros";
    let target = [0xffu8; 32];
    let req = ShallengeRequest {
        username: &user,
        username_len: 10,
        target_hash: &target,
        thread_idx: 0,
        rng_seed: 12345,
    };
    generate_and_check_shallenge(&req)
}

pub fn check_shallenge_hash() -> u32 {
    let expected: [u8; 32] = [
        0xc3, 0x75, 0x0f, 0x87, 0x11, 0xbf, 0x80, 0x9f,
        0x46, 0xde, 0x1f, 0x01, 0xec, 0xeb, 0x6f, 0x4e,
        0x6f, 0xde, 0x67, 0x0a, 0xd8, 0xa3, 0xe2, 0xa6,
        0x00, 0xa0, 0xe0, 0xb7, 0x35, 0x76, 0x54, 0xc9,
    ];
    (shallenge_test().hash == expected) as u32
}

pub fn check_shallenge_nonce_len() -> u32 {
    (shallenge_test().nonce_len == 21) as u32
}

pub fn check_shallenge_is_better() -> u32 {
    shallenge_test().is_better as u32
}

// === compare_hashes (lt / gt / eq branches) ===

pub fn check_compare_hashes_lt() -> u32 {
    let zero = [0u8; 32];
    let max = [0xffu8; 32];
    (compare_hashes(&zero, &max) == -1) as u32
}

pub fn check_compare_hashes_gt() -> u32 {
    let zero = [0u8; 32];
    let max = [0xffu8; 32];
    (compare_hashes(&max, &zero) == 1) as u32
}

pub fn check_compare_hashes_eq() -> u32 {
    let zero = [0u8; 32];
    (compare_hashes(&zero, &zero) == 0) as u32
}

// === Arithmetic primitive bisect (slots 31-40) ===
// The composed primitives above all reduce to the same root cause: any
// integer op that lowers to `mul.hi.u64` (multi-word multiply, divide-by-
// constant via magic-multiply) returns wrong bytes on the current device.
// These slots pin down exactly which PTX op is broken so the alpha-compiler
// regression can be reported against a one-line repro.
//
// Pattern: each `check_arith_*` baselines the expected value via a `const`
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

pub fn check_arith_u32_div_var() -> u32 {
    // Two black-boxed operands — forces `div.u32` PTX op (no magic-multiply
    // folding, since the divisor isn't a known constant).
    const EXPECTED: u32 = ARITH_U32_A / 58;
    let a = core::hint::black_box(ARITH_U32_A);
    let b = core::hint::black_box(58u32);
    (a / b == EXPECTED) as u32
}

pub fn check_arith_u32_div_const() -> u32 {
    // Variable dividend, constant divisor — rustc lowers `x / 58` to
    // `mul.hi.u32` (or `mul.wide.u32` + shift) magic-multiply. Same path
    // base58_encode_32 uses.
    const EXPECTED: u32 = ARITH_U32_A / 58;
    let a = core::hint::black_box(ARITH_U32_A);
    (a / 58 == EXPECTED) as u32
}

pub fn check_arith_u64_div_var() -> u32 {
    // Forces `div.u64` PTX op.
    const EXPECTED: u64 = ARITH_U64_A / 58;
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(58u64);
    (a / b == EXPECTED) as u32
}

pub fn check_arith_u64_div_const() -> u32 {
    // Variable dividend, constant divisor — rustc lowers `x / 58` to
    // `mul.hi.u64` (the smoking-gun op). This is THE path base58_encode_32
    // takes for its divide-by-58 reduction loop.
    const EXPECTED: u64 = ARITH_U64_A / 58;
    let a = core::hint::black_box(ARITH_U64_A);
    (a / 58 == EXPECTED) as u32
}

pub fn check_arith_u32_rem_var() -> u32 {
    // Forces `rem.u32`.
    const EXPECTED: u32 = ARITH_U32_A % 58;
    let a = core::hint::black_box(ARITH_U32_A);
    let b = core::hint::black_box(58u32);
    (a % b == EXPECTED) as u32
}

pub fn check_arith_u64_rem_var() -> u32 {
    // Forces `rem.u64`.
    const EXPECTED: u64 = ARITH_U64_A % 58;
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(58u64);
    (a % b == EXPECTED) as u32
}

pub fn check_arith_u32_mul_lo() -> u32 {
    // Forces `mul.lo.s32` / `mul.lo.u32` (low 32 bits of u32 × u32).
    const EXPECTED: u32 = ARITH_U32_A.wrapping_mul(ARITH_U32_B);
    let a = core::hint::black_box(ARITH_U32_A);
    let b = core::hint::black_box(ARITH_U32_B);
    (a.wrapping_mul(b) == EXPECTED) as u32
}

pub fn check_arith_u64_mul_lo() -> u32 {
    // Forces `mul.lo.s64` / `mul.lo.u64` (low 64 bits of u64 × u64). This
    // op is *heavily* used by the failing primitives but is also used by
    // some passing ones via the slice-indexing path, so it's worth a direct
    // isolated check.
    const EXPECTED: u64 = ARITH_U64_A.wrapping_mul(ARITH_U64_B);
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    (a.wrapping_mul(b) == EXPECTED) as u32
}

pub fn check_arith_u64_mul_hi() -> u32 {
    // The smoking gun: `(a as u128) * (b as u128) >> 64` lowers to a single
    // `mul.hi.u64` PTX op. Every failing primitive (ed25519 field math,
    // secp256k1 field math, base58 divide-by-constant) is dominated by
    // this exact op. If this slot FAILs on GPU and the matching CPU test
    // passes, the alpha compiler's `mul.hi.u64` codegen is broken.
    const PROD: u128 = (ARITH_U64_A as u128) * (ARITH_U64_B as u128);
    const EXPECTED: u64 = (PROD >> 64) as u64;
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    let hi = (((a as u128) * (b as u128)) >> 64) as u64;
    (hi == EXPECTED) as u32
}

pub fn check_arith_u128_mul() -> u32 {
    // Full u128 wrapping multiply. Lowers to a sequence of `mul.lo.s64` +
    // `mul.hi.u64` + `mad.lo.s64`. Exercises the carry chain rustc emits
    // for >64-bit arithmetic.
    const EXPECTED: u128 = ARITH_U128_A.wrapping_mul(ARITH_U128_B);
    let a = core::hint::black_box(ARITH_U128_A);
    let b = core::hint::black_box(ARITH_U128_B);
    (a.wrapping_mul(b) == EXPECTED) as u32
}

// === Composed-primitive sub-bisects (slots 41-45) ===

// 25-byte test vector that lives entirely in the divide-by-58 loop with no
// leading-zero pad. Exercises `base58_encode` (variable length) as opposed
// to `base58_encode_32` (fixed) already covered by slot 3.
const BASE58_VAR_INPUT: [u8; 25] = [
    0x0A, 0xF7, 0x64, 0xC1, 0xB6, 0x13, 0x3A, 0x3A,
    0x0A, 0xBD, 0x7E, 0xF9, 0xC8, 0x53, 0x79, 0x1B,
    0x68, 0x7C, 0xE1, 0xE2, 0x35, 0xF9, 0xDC, 0x84,
    0x66,
];
const BASE58_VAR_EXPECTED: &[u8] = b"5Qw8TAab98QrQmymczzxwkZzacMDL4MeEH";

// Bitcoin Genesis P2PKH (mainnet) — one leading 0x00 forces the
// `num_leading_zeros` pad branch to emit a single '1' before the encoded
// numeric tail.
const BASE58_LEADZERO_INPUT: [u8; 25] = [
    0x00, 0x62, 0xE9, 0x07, 0xB1, 0x5C, 0xBF, 0x27,
    0xD5, 0x42, 0x53, 0x99, 0xEB, 0xF6, 0xF0, 0xFB,
    0x50, 0xEB, 0xB8, 0x8F, 0x18, 0xC2, 0x9B, 0x7D,
    0x93,
];
const BASE58_LEADZERO_EXPECTED: &[u8] = b"1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa";

// 32 all-zero bytes → all-leading-zero pad with no divide loop iterations.
// If this PASSes but slot 3 FAILs, the divide-by-58 codegen is to blame;
// if it FAILs the leading-zero pad logic itself is broken.
const BASE58_ALLZERO_EXPECTED: &[u8] = b"11111111111111111111111111111111";

pub fn check_base58_var_len() -> u32 {
    let mut out = [0u8; 64];
    let n = base58_encode(&BASE58_VAR_INPUT, &mut out);
    if n != BASE58_VAR_EXPECTED.len() {
        return 0;
    }
    let mut i = 0;
    while i < n {
        if out[i] != BASE58_VAR_EXPECTED[i] {
            return 0;
        }
        i += 1;
    }
    1
}

pub fn check_base58_var_len_leading_zero() -> u32 {
    let mut out = [0u8; 64];
    let n = base58_encode(&BASE58_LEADZERO_INPUT, &mut out);
    if n != BASE58_LEADZERO_EXPECTED.len() {
        return 0;
    }
    let mut i = 0;
    while i < n {
        if out[i] != BASE58_LEADZERO_EXPECTED[i] {
            return 0;
        }
        i += 1;
    }
    1
}

pub fn check_base58_all_zeros() -> u32 {
    let input = [0u8; 32];
    let mut out = [0u8; 64];
    let n = base58_encode_32(&input, &mut out);
    if n != BASE58_ALLZERO_EXPECTED.len() {
        return 0;
    }
    let mut i = 0;
    while i < n {
        if out[i] != BASE58_ALLZERO_EXPECTED[i] {
            return 0;
        }
        i += 1;
    }
    1
}

// Captured by running `generate_base64_nonce(0, 12345, &mut [0u8; 21])` on
// the host (see /tmp/probe). Same (thread_idx, rng_seed) the shallenge
// pipeline uses, so this slot directly answers "is the nonce wrong, and is
// that why shallenge_hash fails?".
const XOROSHIRO_NONCE_EXPECTED: [u8; 21] = [
    0x61, 0x63, 0x65, 0x43, 0x48, 0x73, 0x71, 0x46,
    0x36, 0x67, 0x31, 0x33, 0x5a, 0x65, 0x32, 0x6e,
    0x47, 0x53, 0x4a, 0x67, 0x6d,
];

pub fn check_xoroshiro_base64_nonce() -> u32 {
    let mut nonce = [0u8; 21];
    generate_base64_nonce(0, 12345, &mut nonce);
    (nonce == XOROSHIRO_NONCE_EXPECTED) as u32
}

// p2wpkh KAT lifted from bech32::test::should_encode_p2wpkh_correctly.
const BECH32_P2WPKH_HASH: [u8; 20] = [
    0x46, 0x04, 0x7c, 0x8a, 0x3d, 0x8e, 0xdb, 0x13,
    0x4c, 0x3f, 0x1a, 0x3e, 0x7d, 0x65, 0xb0, 0xfd,
    0x74, 0x21, 0xf1, 0x27,
];
const BECH32_P2WPKH_EXPECTED: &[u8] = b"bc1qgcz8ez3a3md3xnplrgl86edsl46zruf8mwx56m";

pub fn check_bech32_p2wpkh() -> u32 {
    let mut out = [0u8; 64];
    let n = encode_p2wpkh_address(&BECH32_P2WPKH_HASH, true, &mut out);
    if n != BECH32_P2WPKH_EXPECTED.len() {
        return 0;
    }
    let mut i = 0;
    while i < n {
        if out[i] != BECH32_P2WPKH_EXPECTED[i] {
            return 0;
        }
        i += 1;
    }
    1
}

// === Tier-2 arithmetic bisect (slots 46-56) ===
// Targets the PTX idioms heavily used by dalek/k256 that the tier-1 net
// (slots 31-40) doesn't directly exercise: carry-chain plumbing, both-lane
// extraction from a single widening mul, fused mul-add, 32×32→64, the
// subtle::Choice mask-blend pattern, and variable shifts.

pub fn check_arith_overflowing_add() -> u32 {
    // Three regimes in one slot so any miscompile of `add.cc.u64` /
    // `addc.cc.u64` (the PTX primitives that carry the boolean out) FAILs
    // the slot regardless of which value range trips it.
    let a = core::hint::black_box(1u64);
    let b = core::hint::black_box(2u64);
    let (s, c) = a.overflowing_add(b);
    if s != 3 || c { return 0; }

    // Carry at the wraparound boundary: u64::MAX + 1 → (0, true)
    let a = core::hint::black_box(u64::MAX);
    let b = core::hint::black_box(1u64);
    let (s, c) = a.overflowing_add(b);
    if s != 0 || !c { return 0; }

    // Saturating-style overflow: u64::MAX + u64::MAX → (u64::MAX-1, true)
    let a = core::hint::black_box(u64::MAX);
    let b = core::hint::black_box(u64::MAX);
    let (s, c) = a.overflowing_add(b);
    if s != u64::MAX - 1 || !c { return 0; }

    1
}

pub fn check_arith_overflowing_sub() -> u32 {
    // No borrow: 5 - 3
    let a = core::hint::black_box(5u64);
    let b = core::hint::black_box(3u64);
    let (s, c) = a.overflowing_sub(b);
    if s != 2 || c { return 0; }

    // Borrow at zero boundary: 0 - 1 → (u64::MAX, true)
    let a = core::hint::black_box(0u64);
    let b = core::hint::black_box(1u64);
    let (s, c) = a.overflowing_sub(b);
    if s != u64::MAX || !c { return 0; }

    // Borrow from mid-range: 1 - u64::MAX → (2, true)
    let a = core::hint::black_box(1u64);
    let b = core::hint::black_box(u64::MAX);
    let (s, c) = a.overflowing_sub(b);
    if s != 2 || !c { return 0; }

    1
}

pub fn check_arith_carry_chain_3limb() -> u32 {
    // Three-limb add chosen so the carry propagates through every limb.
    // [u64::MAX, u64::MAX, 0] + [1, 0, 0] = [0, 0, 1]
    // This is the literal shape dalek's Scalar52::add / k256's
    // FieldElement::add expand to, so a miscompile of the carry-propagation
    // PTX sequence (overflowing_add + boolean OR + add of `prev_carry as
    // u64`) corrupts every field-element add silently.
    let a0 = core::hint::black_box(u64::MAX);
    let a1 = core::hint::black_box(u64::MAX);
    let a2 = core::hint::black_box(0u64);
    let b0 = core::hint::black_box(1u64);
    let b1 = core::hint::black_box(0u64);
    let b2 = core::hint::black_box(0u64);

    let (s0, c0) = a0.overflowing_add(b0);
    let (s1a, c1a) = a1.overflowing_add(b1);
    let (s1, c1b) = s1a.overflowing_add(c0 as u64);
    let c1 = c1a | c1b;
    let (s2a, c2a) = a2.overflowing_add(b2);
    let (s2, c2b) = s2a.overflowing_add(c1 as u64);
    let _c2 = c2a | c2b;

    (s0 == 0 && s1 == 0 && s2 == 1) as u32
}

pub fn check_arith_widening_mul_pair() -> u32 {
    // Tier-1 slots 38 (lo) and 39 (hi) verify each lane in isolation. This
    // one verifies both lanes come from the *same* widening product, the
    // way schoolbook multiplies in dalek/k256 consume them. A bug that
    // swaps lane association passes 38 and 39 individually but FAILs here.
    const PROD: u128 = (ARITH_U64_A as u128) * (ARITH_U64_B as u128);
    const EXPECTED_LO: u64 = PROD as u64;
    const EXPECTED_HI: u64 = (PROD >> 64) as u64;

    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    let p = (a as u128) * (b as u128);
    let lo = p as u64;
    let hi = (p >> 64) as u64;
    (lo == EXPECTED_LO && hi == EXPECTED_HI) as u32
}

pub fn check_arith_mad_lo_u64() -> u32 {
    // `a.wrapping_mul(b).wrapping_add(c)` typically folds to a single
    // `mad.lo.u64` PTX op. This is dalek's `m!` macro shape — slot 38 only
    // tests the mul, so a MAD-folding-only codegen bug slips through.
    const EXPECTED: u64 = ARITH_U64_A.wrapping_mul(ARITH_U64_B).wrapping_add(ARITH_U64_A);
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    let c = core::hint::black_box(ARITH_U64_A);
    (a.wrapping_mul(b).wrapping_add(c) == EXPECTED) as u32
}

pub fn check_arith_mad_hi_u64() -> u32 {
    // High-half MAD: same shape as mad_lo but pulling the upper 64 bits
    // of the widening product before the add. May fold to `mad.hi.u64`.
    const PROD: u128 = (ARITH_U64_A as u128) * (ARITH_U64_B as u128);
    const EXPECTED: u64 = ((PROD >> 64) as u64).wrapping_add(ARITH_U64_A);

    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    let c = core::hint::black_box(ARITH_U64_A);
    let hi = (((a as u128) * (b as u128)) >> 64) as u64;
    (hi.wrapping_add(c) == EXPECTED) as u32
}

pub fn check_arith_mul_wide_u32() -> u32 {
    // Both operands start as u32 then widen to u64 for the mul — rustc may
    // emit `mul.wide.u32` (one PTX op, distinct from `mul.lo.u64`). k256's
    // 32-bit big-int paths take exactly this shape.
    const EXPECTED: u64 = (ARITH_U32_A as u64) * (ARITH_U32_B as u64);
    let a = core::hint::black_box(ARITH_U32_A);
    let b = core::hint::black_box(ARITH_U32_B);
    ((a as u64) * (b as u64) == EXPECTED) as u32
}

pub fn check_arith_mask_blend_true() -> u32 {
    // The subtle::Choice / CtOption idiom: bool → u64 → wrapping_neg gives
    // an all-1s or all-0s mask; (a & mask) | (b & !mask) selects a or b.
    // k256's `SecretKey::from_bytes(...).unwrap()` runs through CtOption
    // whose unwrap is a const-time select on the validity flag — if the
    // `cond as u64` → `wrapping_neg()` lowering is wrong, unwrap silently
    // returns the wrong arm (matches the "consistent-but-wrong" secp256k1
    // symptom).
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    let cond = core::hint::black_box(true);
    let mask = (cond as u64).wrapping_neg();
    let r = (a & mask) | (b & !mask);
    (r == ARITH_U64_A) as u32
}

pub fn check_arith_mask_blend_false() -> u32 {
    // Same as above but with cond=false — selects b. Splitting true/false
    // into two slots means a bug that breaks only one arm pinpoints
    // immediately.
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    let cond = core::hint::black_box(false);
    let mask = (cond as u64).wrapping_neg();
    let r = (a & mask) | (b & !mask);
    (r == ARITH_U64_B) as u32
}

pub fn check_arith_var_shr_u64() -> u32 {
    // Runtime shift amount — emits `shr.b64 %rd, %rd, %r` (variable form),
    // distinct from constant-amount shifts which can be folded. Montgomery
    // reductions in k256 do variable shifts during scalar splitting.
    const EXPECTED: u64 = ARITH_U64_A >> 13;
    let a = core::hint::black_box(ARITH_U64_A);
    let n = core::hint::black_box(13u32);
    (a >> n == EXPECTED) as u32
}

pub fn check_arith_var_shl_u64() -> u32 {
    // Same as var_shr but the other direction (`shl.b64`).
    const EXPECTED: u64 = ARITH_U64_A << 13;
    let a = core::hint::black_box(ARITH_U64_A);
    let n = core::hint::black_box(13u32);
    (a << n == EXPECTED) as u32
}

pub fn check_arith_blackbox_identity_u64() -> u32 {
    // The cheapest possible probe: does black_box preserve a u64?
    // No arithmetic of any kind — if this FAILs on GPU + PASSes on CPU,
    // black_box's PTX lowering doesn't preserve the value, and every
    // tier-1/tier-2 arith slot's FAIL is a black_box artifact, not an op
    // bug. Tests with `0xDEADBEEFCAFEBABE` so a zero return is obviously
    // wrong.
    let v: u64 = 0xDEADBEEFCAFEBABE;
    (core::hint::black_box(v) == v) as u32
}

pub fn check_arith_blackbox_identity_u32() -> u32 {
    // u32 variant — same probe at half the width in case the bug is
    // type-specific.
    let v: u32 = 0xDEADBEEF;
    (core::hint::black_box(v) == v) as u32
}

pub fn check_base58_div_by_58() -> u32 {
    // Exact divmod-by-58 pattern from base58_encode_32's digit-extraction
    // loop (logic/src/base58.rs:73-82), in isolation. The constant-divisor
    // `/ 58^k` and `% 58` lowerings emit `mul.hi.u64` magic-multiply ops
    // in PTX — the smoking-gun op from earlier inspection. No alphabet
    // lookup, no output[] writes, no leading-zero pad, no dynamic loops —
    // just the arithmetic that produces digit values in [0, 58).
    //
    // If this FAILs on GPU while the surrounding non-arithmetic code (sha2,
    // ripemd, etc.) PASSes, slot 41's `Invalid __global__ read` cascade
    // is downstream of broken div/mod codegen (corrupted digit values →
    // garbage byte writes → corrupted state → wild address used as an
    // alphabet index), not an independent OOB bug.
    const LIMB: u64 = 0x0123_4567_89AB_CDEF;
    const EXPECTED: [u8; 5] = [
        ((LIMB / 1) % 58) as u8,
        ((LIMB / 58) % 58) as u8,
        ((LIMB / (58 * 58)) % 58) as u8,
        ((LIMB / (58 * 58 * 58)) % 58) as u8,
        ((LIMB / (58 * 58 * 58 * 58)) % 58) as u8,
    ];

    let limb = core::hint::black_box(LIMB);
    let got: [u8; 5] = [
        ((limb / 1) % 58) as u8,
        ((limb / 58) % 58) as u8,
        ((limb / (58 * 58)) % 58) as u8,
        ((limb / (58 * 58 * 58)) % 58) as u8,
        ((limb / (58 * 58 * 58 * 58)) % 58) as u8,
    ];

    (got == EXPECTED) as u32
}

pub fn check_iter_static_table_lookup() -> u32 {
    // Simplest possible probe for `TABLE[byte as usize]`: a single dynamic
    // index into a small static byte slice. No iterator, no &mut, no slice
    // projection — pure indexed read from a `&'static [u8; N]` plus an
    // equality check.
    const TABLE: &[u8; 4] = b"ABCD";
    let idx = core::hint::black_box(0usize);
    (TABLE[idx] == b'A') as u32
}

pub fn check_iter_mut_slice_partial() -> u32 {
    // `for val in &mut buf[..n]` over a partial slice of a stack-resident
    // fixed-size array, writing a constant. Isolates the IterMut codegen
    // from any table lookup. If this FAILs, the iter_mut over a sliced
    // `&mut [T; N]` is the broken op.
    let mut buf = [0u8; 8];
    let n = core::hint::black_box(4usize);
    for val in &mut buf[..n] {
        *val = 0xAA;
    }
    (buf[0] == 0xAA && buf[3] == 0xAA && buf[4] == 0 && buf[7] == 0) as u32
}

pub fn check_iter_mut_alphabet_lookup() -> u32 {
    // Combined: `for val in &mut buf[..n] { *val = TABLE[*val as usize]; }`
    // — the exact final-stage pattern in base58_encode_32 that runs even
    // when the divide loop is dead (slot 43's failure case). Mirror of
    // base58.rs:99-101.
    const TABLE: &[u8; 4] = b"ABCD";
    let mut buf = [0u8; 8];
    let n = core::hint::black_box(4usize);
    for val in &mut buf[..n] {
        *val = TABLE[*val as usize];
    }
    (buf[0] == b'A' && buf[3] == b'A' && buf[4] == 0 && buf[7] == 0) as u32
}

pub fn check_iter_static_slice_lookup() -> u32 {
    // Counterpart to slot 60. Identical shape — single dynamic index into
    // a small static byte table — but typed as `&'static [u8]` (slice)
    // instead of `&'static [u8; 4]` (array reference). Slot 44 already
    // hinted that slice-typed alphabets don't crash where array-ref-typed
    // ones do (compare xoroshiro `&[u8]` → FAIL no crash, vs base58/
    // bech32 `&[u8; N]` → CRASH). This slot makes the discriminator a
    // controlled one-variable test: if 60 CRASHes and 63 PASSes, the
    // backend mishandles array-ref static indexing specifically.
    const TABLE: &[u8] = b"ABCD";
    let idx = core::hint::black_box(0usize);
    (TABLE[idx] == b'A') as u32
}

pub fn check_arith_divrem_by_58_pow_5() -> u32 {
    // Slot 59 covers `x / 58` through `x / 58^4`. base58_encode_32's limb
    // update loop divides by `58^5 = 656_356_768` (NEXT_LIMB_DIVISOR),
    // which produces a *different* magic-multiply constant than slot 59
    // exercises (PTX inspection of v1.42 confirms: 7_544_311_872_078_572_213
    // for /58^5, distinct from slot 59's constants). This slot covers
    // exactly that gap.
    //
    // Cases include the first 4 bytes of slot 3's input, the divisor
    // itself, divisor-1 boundary, a known quotient with non-zero
    // remainder, and u64 extremes.
    //
    // Layout note: parallel primitive arrays (INPUTS / EXPECTED_Q /
    // EXPECTED_R) instead of an `[(u64, u64, u64); N]` array of tuples.
    // cuda-oxide as of 5feaf2e doesn't handle tuple-element array
    // constants (`translate_array_constant` only takes the integer-element
    // branch); parallel arrays of primitives go through the supported
    // path. Semantics are identical to the tuple form.
    const D: u64 = 58_u64.pow(5);
    const INPUTS: [u64; 6] = [
        0x089A23FF,
        D,
        D - 1,
        D.wrapping_mul(7).wrapping_add(123),
        u64::MAX,
        0xFFFFFFFF_00000000,
    ];
    const EXPECTED_Q: [u64; 6] = [
        0x089A23FF_u64 / D,
        1,
        0,
        7,
        u64::MAX / D,
        0xFFFFFFFF_00000000_u64 / D,
    ];
    const EXPECTED_R: [u64; 6] = [
        0x089A23FF_u64 % D,
        0,
        D - 1,
        123,
        u64::MAX % D,
        0xFFFFFFFF_00000000_u64 % D,
    ];

    let mut i = 0;
    while i < INPUTS.len() {
        let x = core::hint::black_box(INPUTS[i]);
        if x / D != EXPECTED_Q[i] || x % D != EXPECTED_R[i] {
            return 0;
        }
        i += 1;
    }
    1
}

pub fn check_arith_i128_chain_add() -> u32 {
    // Slot 40 (u128 wrapping_mul) and slot 49 (widening mul pair) both
    // PASS, but those only exercise a single u128 op. dalek's
    // Scalar52::mul_internal and k256's FieldElement5x52::mul_inner
    // accumulate ~25 widening products via sequential `u128 + u128`
    // chains. Each addition must propagate the low-half carry into the
    // high half. If that's broken, every accumulation step silently
    // drops a bit — which would explain the residual failure of slots
    // 2/4/5/11–20 even after the overflowing_add fix.
    //
    // Three regimes per the cuda-oxide divrem_large_const_repro: pure
    // low→high carry, carry-rolls-fully-over (both halves saturated),
    // and combined low+high adds.

    // Case 0: (MAX, 0) + (1, 0) = (0, 1). Pure low→high carry.
    {
        let a = core::hint::black_box(u64::MAX as u128);
        let b = core::hint::black_box(1u128);
        const E: u128 = (u64::MAX as u128).wrapping_add(1);
        if a.wrapping_add(b) != E {
            return 0;
        }
    }

    // Case 1: (MAX, MAX) + (1, 0) = (0, 0). Carry rolls all the way over.
    {
        let a = core::hint::black_box(u128::MAX);
        let b = core::hint::black_box(1u128);
        const E: u128 = u128::MAX.wrapping_add(1);
        if a.wrapping_add(b) != E {
            return 0;
        }
    }

    // Case 2: 4-operand chain forcing carry on every step.
    // (MAX_LO) + (MAX_LO) + (MAX_LO) + ((1 << 64) | 1)
    //   → low halves wrap three times (3 carries to high) + 1 from d's
    //     high half → high = 4, low = ((MAX*3) wrapping) + 1.
    {
        let a = core::hint::black_box(u64::MAX as u128);
        let b = core::hint::black_box(u64::MAX as u128);
        let c = core::hint::black_box(u64::MAX as u128);
        let d = core::hint::black_box((1u128 << 64) | 1u128);
        let s = a.wrapping_add(b).wrapping_add(c).wrapping_add(d);
        const E: u128 = (u64::MAX as u128)
            .wrapping_add(u64::MAX as u128)
            .wrapping_add(u64::MAX as u128)
            .wrapping_add((1u128 << 64) | 1u128);
        if s != E {
            return 0;
        }
    }

    1
}

pub fn check_base58_limb_divrem() -> u32 {
    // The exact base58_encode_32 inner-loop shape: a u32 limb loaded
    // from a stack array, shifted into the high half, added to a u64
    // carry, then div/mod by NEXT_LIMB_DIVISOR. Slot 64 covers div by
    // 58^5 from a clean u64 source; this slot covers the case where
    // the multiplicand goes into `mul.hi.u64` after being reconstructed
    // via `shl + add`. The discriminator is the operand path, not the
    // divisor.
    //
    // Runtime index defeats mem2reg so `limbs[]` actually lives in
    // local memory and the read materialises as ld.local.b32. Even if
    // the optimizer tracks the value across the local store, the
    // multiplicand `%dividend` still comes from `add.s64(carry, shl.b64(limb,
    // 32))` — that's the suspect shape.
    const D: u64 = 58_u64.pow(5);
    let mut limbs = [0u32; 8];
    let write_idx = core::hint::black_box(3usize) & 7;
    let limb_val: u32 = core::hint::black_box(0x089A_23FF_u32);
    limbs[write_idx] = limb_val;

    let carry: u64 = core::hint::black_box(0xDEAD_BEEF_u64);
    let dividend = carry.wrapping_add((limbs[write_idx] as u64) << 32);

    // Const-eval baseline computed on the host rustc.
    const EXPECTED_DIVIDEND: u64 = 0xDEAD_BEEF_u64.wrapping_add((0x089A_23FF_u64) << 32);
    const EXPECTED_Q: u64 = EXPECTED_DIVIDEND / D;
    const EXPECTED_R: u64 = EXPECTED_DIVIDEND % D;

    (dividend == EXPECTED_DIVIDEND
        && dividend / D == EXPECTED_Q
        && dividend % D == EXPECTED_R) as u32
}

pub fn check_dynamic_index_write() -> u32 {
    // base58_encode_32's dynamic-growth pattern in isolation:
    //   while remaining_carry > 0 && limb_count < N {
    //       limbs[limb_count] = (remaining_carry % D) as u32;
    //       remaining_carry /= D;
    //       limb_count += 1;
    //   }
    // Slot 43 ([0u8; 32] input) PASSes because this loop never runs
    // with non-zero values; slot 3 (real input) FAILs and exercises
    // it heavily. Slot 60-62 cover runtime-index *reads*; this one is
    // about *writes* where the index variable mutates across loop
    // iterations.
    //
    // Verifies all 10 slots of the resulting array against the
    // host-CPU baseline computed under `const`.
    const D: u64 = 58_u64.pow(5);
    let mut limbs = [0u32; 10];
    let mut limb_count: usize = 0;
    let mut remaining_carry = core::hint::black_box(0xDEAD_BEEF_CAFE_BABE_u64);

    while remaining_carry > 0 && limb_count < 10 {
        limbs[limb_count] = (remaining_carry % D) as u32;
        remaining_carry /= D;
        limb_count += 1;
    }

    // Host-side const-eval of the same loop. Split into two parallel
    // const fns (one returning the array, one returning the count)
    // because cuda-oxide v1.43 can't lower tuple constants whose fields
    // are arrays — same limitation that forced the parallel-primitive
    // arrays layout in slot 64.
    const fn run_growth_limbs() -> [u32; 10] {
        let mut out = [0u32; 10];
        let mut count = 0usize;
        let mut c = 0xDEAD_BEEF_CAFE_BABE_u64;
        while c > 0 && count < 10 {
            out[count] = (c % D) as u32;
            c /= D;
            count += 1;
        }
        out
    }
    const fn run_growth_count() -> usize {
        let mut count = 0usize;
        let mut c = 0xDEAD_BEEF_CAFE_BABE_u64;
        while c > 0 && count < 10 {
            c /= D;
            count += 1;
        }
        count
    }
    const EXPECTED_LIMBS: [u32; 10] = run_growth_limbs();
    const EXPECTED_COUNT: usize = run_growth_count();

    if limb_count != EXPECTED_COUNT {
        return 0;
    }
    let mut i = 0;
    while i < 10 {
        if limbs[i] != EXPECTED_LIMBS[i] {
            return 0;
        }
        i += 1;
    }
    1
}

pub fn check_arith_widening_mul_chain_3term() -> u32 {
    // dalek's `Scalar52::mul_internal` and k256's
    // `FieldElement5x52::mul_inner` accumulate partial products as
    //   z = m(a0, b2) + m(a1, b1) + m(a2, b0)
    // where m(x, y) = `(x as u128) * (y as u128)`. Slot 40 (one u128
    // wrapping_mul) and slot 49 (one widening pair) both PASS; slot
    // 65 covers a chain of u128 adds. This composes them: three
    // widening mults summed via `u128 + u128 + u128`. If a register-
    // pressure or scheduling bug only surfaces under composition,
    // this slot catches it where the isolated ones don't.
    //
    // 52-bit operands (the actual limb size dalek uses) → products
    // span ~104 bits, sums span ~106, forcing real carries across
    // the 64-bit boundary in every step.
    const A0: u64 = 0x000F_FFFF_FFFF_FFFF;
    const A1: u64 = 0x000F_FFFF_FFFF_FFFE;
    const A2: u64 = 0x000F_FFFF_FFFF_FFFD;
    const B0: u64 = 0x000F_FFFF_FFFF_FFFC;
    const B1: u64 = 0x000F_FFFF_FFFF_FFFB;
    const B2: u64 = 0x000F_FFFF_FFFF_FFFA;
    const EXPECTED: u128 = (A0 as u128).wrapping_mul(B2 as u128)
        .wrapping_add((A1 as u128).wrapping_mul(B1 as u128))
        .wrapping_add((A2 as u128).wrapping_mul(B0 as u128));

    let a0 = core::hint::black_box(A0) as u128;
    let a1 = core::hint::black_box(A1) as u128;
    let a2 = core::hint::black_box(A2) as u128;
    let b0 = core::hint::black_box(B0) as u128;
    let b1 = core::hint::black_box(B1) as u128;
    let b2 = core::hint::black_box(B2) as u128;

    let z = a0.wrapping_mul(b2)
        .wrapping_add(a1.wrapping_mul(b1))
        .wrapping_add(a2.wrapping_mul(b0));
    (z == EXPECTED) as u32
}

// Slot 69: base58 Phase A inner-mutate phase in isolation.
//
// `base58_encode_32` runs an 8-iter outer loop. Each iter does two phases:
//   Phase A — `for i in 0..limb_count { rc += (limbs[i] << 32); limbs[i] = (rc % D) as u32; rc /= D; }`
//   Phase B — `if rc > 0 && limb_count < 10 { limbs[limb_count] = ...; limb_count += 1; }` (×2)
// Slot 67 covered Phase B. Slot 43 (all-zero input) PASSes because Phase A
// never runs (limb_count stays 0). Slot 41 (non-zero input) FAILs and the
// only path it touches that slot 43 doesn't is Phase A. This slot runs
// Phase A standalone with limb_count=1 and a non-zero limb so the loop
// executes exactly one iteration of read-shift-add-divrem-writeback.
pub fn check_base58_inner_mutate_phase() -> u32 {
    const D: u64 = 58_u64.pow(5);
    const LIMB0_IN: u32 = 0x1234_5678;
    const CHUNK_IN: u32 = 0x89AB_CDEF;
    // Const-eval baseline (same loop body, fully evaluated at compile time).
    const fn expected_limb0() -> u32 {
        let rc: u64 = (CHUNK_IN as u64) + ((LIMB0_IN as u64) << 32);
        (rc % D) as u32
    }
    const fn expected_rc() -> u64 {
        let rc: u64 = (CHUNK_IN as u64) + ((LIMB0_IN as u64) << 32);
        rc / D
    }
    const E_LIMB0: u32 = expected_limb0();
    const E_RC: u64 = expected_rc();

    let mut limbs = [0u32; 10];
    limbs[0] = core::hint::black_box(LIMB0_IN);
    let limb_count: usize = core::hint::black_box(1);

    let chunk: u32 = core::hint::black_box(CHUNK_IN);
    let mut remaining_carry: u64 = chunk as u64;
    for i in 0..limb_count {
        remaining_carry += (limbs[i] as u64) << 32;
        limbs[i] = (remaining_carry % D) as u32;
        remaining_carry /= D;
    }

    (limbs[0] == E_LIMB0 && remaining_carry == E_RC) as u32
}

// Slot 70: curve25519-dalek `clamp_integer` in isolation. The smallest
// possible dalek call — pure bit-mask on bytes 0 and 31, no field math,
// no scalar repr conversion. If THIS fails, the bug is in the dalek
// API-entry plumbing itself, not in any arithmetic.
//
// Clamp definition (RFC 7748 / dalek):
//   byte[0]  &= 0xF8        (clear bits 0,1,2)
//   byte[31] &= 0x7F        (clear bit 7)
//   byte[31] |= 0x40        (set bit 6)
pub fn check_dalek_clamp_integer() -> u32 {
    let input: [u8; 32] = [0xFF; 32];
    let clamped = curve25519_dalek::scalar::clamp_integer(input);
    const EXPECTED: [u8; 32] = {
        let mut e = [0xFFu8; 32];
        e[0] = 0xF8;
        e[31] = 0x7F; // (0xFF & 0x7F) | 0x40 = 0x7F
        e
    };
    (clamped == EXPECTED) as u32
}

// Slot 71: `Scalar::from_bytes_mod_order` round-trip for the canonical
// scalar 1 (little-endian [1, 0, …, 0]). 1 is already < l, so reduction
// is a no-op and `to_bytes()` must return the input unchanged. Tests
// Scalar52 deserialization + canonical encoding without exercising any
// field multiplication or scalar mul.
pub fn check_dalek_scalar_round_trip_one() -> u32 {
    let mut input = [0u8; 32];
    input[0] = 1;
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(input);
    let bytes = scalar.to_bytes();
    (bytes == input) as u32
}

// Slot 72: `EdwardsPoint::mul_base(scalar=1).compress()` must equal the
// well-known ed25519 basepoint encoding (RFC 8032). Exercises the full
// fixed-base scalar-mult path with the smallest non-trivial scalar.
//
// If slots 70/71 PASS and 72 FAILs, the bug is in mul_base / EdwardsPoint
// ops or in `compress()` (field inversion), not in the scalar plumbing.
const ED25519_BASEPOINT_COMPRESSED: [u8; 32] = [
    0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
];

pub fn check_dalek_mul_base_scalar_one() -> u32 {
    let mut scalar_bytes = [0u8; 32];
    scalar_bytes[0] = 1;
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(scalar_bytes);
    let point = curve25519_dalek::EdwardsPoint::mul_base(&scalar);
    let compressed = point.compress().to_bytes();
    (compressed == ED25519_BASEPOINT_COMPRESSED) as u32
}

// Slot 73: `SecretKey::from_bytes` for the smallest valid scalar (=1).
// Tests just the validation/wrap step (range check + GenericArray copy).
// k256 scalars are big-endian, so 1 = [0; 31] ++ [0x01].
//
// Wrapped in ManuallyDrop because SecretKey zeroizes on Drop and
// cuda-oxide does not yet emit device-side drop_in_place (same pattern
// as logic/src/secp256k1.rs).
pub fn check_k256_secret_from_bytes_one() -> u32 {
    use core::mem::ManuallyDrop;
    use k256::SecretKey;
    let mut priv_bytes = [0u8; 32];
    priv_bytes[31] = 1;
    let result = SecretKey::from_bytes((&priv_bytes).into());
    match result {
        Ok(sk) => {
            let _sk = ManuallyDrop::new(sk);
            1
        }
        Err(_) => 0,
    }
}

// Slot 74: full k256 derive for scalar=1. Compressed public key must
// equal the well-known secp256k1 generator G.
const SECP256K1_GENERATOR_COMPRESSED: [u8; 33] = [
    0x02, 0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB,
    0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
    0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28,
    0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17,
    0x98,
];

pub fn check_k256_derive_scalar_one() -> u32 {
    let mut priv_bytes = [0u8; 32];
    priv_bytes[31] = 1;
    let pub_key = secp256k1_derive_public_key(&priv_bytes);
    (pub_key == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 75: full k256 derive for scalar=2. Compressed public key must
// equal 2G (one more doubling beyond slot 74). A 74-PASS / 75-FAIL split
// pinpoints the doubling formula; a 74-FAIL / 75-FAIL means scalar mult
// is broken even for the trivial-scalar case.
const SECP256K1_TWO_G_COMPRESSED: [u8; 33] = [
    0x02, 0xC6, 0x04, 0x7F, 0x94, 0x41, 0xED, 0x7D,
    0x6D, 0x30, 0x45, 0x40, 0x6E, 0x95, 0xC0, 0x7C,
    0xD8, 0x5C, 0x77, 0x8E, 0x4B, 0x8C, 0xEF, 0x3C,
    0xA7, 0xAB, 0xAC, 0x09, 0xB9, 0x5C, 0x70, 0x9E,
    0xE5,
];

pub fn check_k256_derive_scalar_two() -> u32 {
    let mut priv_bytes = [0u8; 32];
    priv_bytes[31] = 2;
    let pub_key = secp256k1_derive_public_key(&priv_bytes);
    (pub_key == SECP256K1_TWO_G_COMPRESSED) as u32
}

// Slot 76: bare `&'static [u64; 5]` runtime-indexed read. The simplest
// possible test of the "element-width > 1 byte breaks &'static reads"
// hypothesis. No struct wrapper, no arithmetic on the result.
static STATIC_U64_TABLE: [u64; 5] = [
    0x0123_4567_89AB_CDEF,
    0xFEDC_BA98_7654_3210,
    0x1111_2222_3333_4444,
    0xAAAA_BBBB_CCCC_DDDD,
    0xDEAD_BEEF_CAFE_BABE,
];

pub fn check_static_u64_array_lookup() -> u32 {
    let idx = core::hint::black_box(3usize);
    let val = STATIC_U64_TABLE[idx];
    (val == 0xAAAA_BBBB_CCCC_DDDD) as u32
}

// Slot 77: same but wrapped in a single-field tuple struct — matches
// dalek's `Scalar52(pub(crate) [u64; 5])` newtype shape. If 76 PASSes
// and 77 FAILs, the bug is specifically in field projection through a
// newtype, not in the underlying array.
#[repr(transparent)]
pub struct U64Wrap5(pub [u64; 5]);

static STATIC_U64_WRAPPED: U64Wrap5 = U64Wrap5([
    0x0123_4567_89AB_CDEF,
    0xFEDC_BA98_7654_3210,
    0x1111_2222_3333_4444,
    0xAAAA_BBBB_CCCC_DDDD,
    0xDEAD_BEEF_CAFE_BABE,
]);

pub fn check_static_struct_wrapped_u64_lookup() -> u32 {
    let idx = core::hint::black_box(3usize);
    let val = STATIC_U64_WRAPPED.0[idx];
    (val == 0xAAAA_BBBB_CCCC_DDDD) as u32
}

// Slot 78: encode the secp256k1 generator point directly — no scalar mult,
// no Lazy<> table touch. Tests the projective→affine + to_encoded_point
// chain in isolation. ProjectivePoint::GENERATOR has z=1, so the affine
// conversion's field inversion is trivial; this primarily exercises the
// FieldElement→bytes serialization + parity-bit pack.
pub fn check_k256_encode_generator() -> u32 {
    use k256::ProjectivePoint;
    use k256::elliptic_curve::sec1::ToEncodedPoint;
    let g = ProjectivePoint::GENERATOR;
    let affine = g.to_affine();
    let encoded = affine.to_encoded_point(true);
    let bytes = encoded.as_bytes();
    if bytes.len() != 33 {
        return 0;
    }
    let mut out = [0u8; 33];
    out.copy_from_slice(bytes);
    (out == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 79: `ProjectivePoint::double()` on the generator + encode. One
// doubling = one field-mul-heavy operation that produces a projective
// point with z != 1, so the subsequent `to_affine()` requires a real
// field inversion. 78 PASS + 79 FAIL = doubling formula or non-trivial
// field inversion broken (5-wide variant of Bug C suspect).
pub fn check_k256_double_generator() -> u32 {
    use k256::ProjectivePoint;
    use k256::elliptic_curve::sec1::ToEncodedPoint;
    let g2 = ProjectivePoint::GENERATOR.double();
    let affine = g2.to_affine();
    let encoded = affine.to_encoded_point(true);
    let bytes = encoded.as_bytes();
    if bytes.len() != 33 {
        return 0;
    }
    let mut out = [0u8; 33];
    out.copy_from_slice(bytes);
    (out == SECP256K1_TWO_G_COMPRESSED) as u32
}

// Slot 80: k256 `Scalar::ONE` round-trip via the PrimeField trait. Mirror
// of slot 71 for k256's Scalar type. k256's Scalar wraps a `U256` from
// crypto-bigint (different layout than dalek's `Scalar52([u64; 5])`),
// so this distinguishes Bug A (dalek-specific newtype shape) from a
// broader Bug A' (any static-resident scalar repr).
pub fn check_k256_scalar_one_round_trip() -> u32 {
    use k256::Scalar;
    use k256::elliptic_curve::PrimeField;
    let s = Scalar::ONE;
    let repr = s.to_repr();
    let s2_opt = Scalar::from_repr(repr);
    let recovered: bool = s2_opt.is_some().into();
    if !recovered {
        return 0;
    }
    let s2 = s2_opt.unwrap();
    (s2 == s) as u32
}

// Slot 81: `u128 >> 52` immediate right shift, matching the exact shape
// inside dalek's `montgomery_reduce::part1`:
//   ((sum + m(p, constants::L[0])) >> 52, p)
// LLVM lowers u128 immediate shifts to multi-step 64-bit shift sequences.
// Slot 65 (i128 add chain) is fixed but doesn't cover this shape. Slot
// 55/56 cover u64 var shifts, not u128 immediate shifts.
pub fn check_arith_u128_imm_shr_52() -> u32 {
    const SUM: u128 = 0xFEDC_BA98_7654_3210_0123_4567_89AB_CDEF;
    const EXPECTED: u128 = SUM >> 52;
    let sum = core::hint::black_box(SUM);
    let shifted = sum >> 52;
    (shifted == EXPECTED) as u32
}

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

pub fn check_static_depth4_newtype_nesting() -> u32 {
    let idx = core::hint::black_box(2usize);
    let v = NESTED_ONE_PROBE.0.limbs[idx].0;
    (v == 0x9999_AAAA_BBBB_CCCC) as u32
}

// Slot 83: reverse range iterator `(0..N).rev()` writing into a stack
// array. The only loop shape inside base58_encode_32's digit-extraction
// phase that isn't covered by an existing isolated slot:
//   for idx in (0..limb_count).rev() {
//       let output_offset = idx * DIGITS_PER_LIMB;
//       output[output_offset + i] = ...;
//   }
pub fn check_reverse_range_write() -> u32 {
    let limb_count: usize = core::hint::black_box(3);
    let mut out = [0u32; 10];
    for idx in (0..limb_count).rev() {
        out[idx] = (idx as u32) * 100;
    }
    const fn expected() -> [u32; 10] {
        let mut e = [0u32; 10];
        let mut idx = 3usize;
        while idx > 0 {
            idx -= 1;
            e[idx] = (idx as u32) * 100;
        }
        e
    }
    const EXPECTED: [u32; 10] = expected();
    (out == EXPECTED) as u32
}

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
mod bisect_scalar52 {
    //! Verbatim copy of the parts of `curve25519_dalek::backend::serial::u64::scalar`
    //! we need for slot 71's ladder bisect. If the GPU produces wrong
    //! results from THIS code (which is identical to what dalek runs
    //! internally), the bug is in the compiler's lowering of these
    //! specific Rust idioms, not in dalek's API surface.
    //!
    //! Source: curve25519-dalek 4.1.3.

    /// u64 * u64 = u128 multiply helper (dalek's `m`).
    #[inline(always)]
    fn m(x: u64, y: u64) -> u128 {
        (x as u128) * (y as u128)
    }

    #[derive(Copy, Clone)]
    pub struct Scalar52(pub [u64; 5]);

    /// `L` = group order of curve25519's scalar field.
    pub const L: Scalar52 = Scalar52([
        0x0002631a5cf5d3ed,
        0x000dea2f79cd6581,
        0x000000000014def9,
        0x0000000000000000,
        0x0000100000000000,
    ]);

    /// `L` * LFACTOR ≡ -1 (mod 2^52)
    pub const LFACTOR: u64 = 0x51da312547e1b;

    /// `R` = 2^260 mod L
    pub const R: Scalar52 = Scalar52([
        0x000f48bd6721e6ed,
        0x0003bab5ac67e45a,
        0x000fffffeb35e51b,
        0x000fffffffffffff,
        0x00000fffffffffff,
    ]);

    impl Scalar52 {
        pub const ZERO: Scalar52 = Scalar52([0, 0, 0, 0, 0]);

        /// Unpack 32 bytes (little-endian) into 5 52-bit limbs.
        pub fn from_bytes(bytes: &[u8; 32]) -> Scalar52 {
            let mut words = [0u64; 4];
            for i in 0..4 {
                for j in 0..8 {
                    words[i] |= (bytes[(i * 8) + j] as u64) << (j * 8);
                }
            }
            let mask = (1u64 << 52) - 1;
            let top_mask = (1u64 << 48) - 1;
            let mut s = Scalar52::ZERO;
            s.0[0] =   words[0]                            & mask;
            s.0[1] = ((words[0] >> 52) | (words[1] << 12)) & mask;
            s.0[2] = ((words[1] >> 40) | (words[2] << 24)) & mask;
            s.0[3] = ((words[2] >> 28) | (words[3] << 36)) & mask;
            s.0[4] =  (words[3] >> 16)                     & top_mask;
            s
        }

        /// Pack 5 52-bit limbs into 32 bytes (little-endian).
        #[allow(clippy::identity_op)]
        pub fn as_bytes(&self) -> [u8; 32] {
            let mut s = [0u8; 32];
            s[ 0] =  (self.0[ 0] >>  0)                      as u8;
            s[ 1] =  (self.0[ 0] >>  8)                      as u8;
            s[ 2] =  (self.0[ 0] >> 16)                      as u8;
            s[ 3] =  (self.0[ 0] >> 24)                      as u8;
            s[ 4] =  (self.0[ 0] >> 32)                      as u8;
            s[ 5] =  (self.0[ 0] >> 40)                      as u8;
            s[ 6] = ((self.0[ 0] >> 48) | (self.0[ 1] << 4)) as u8;
            s[ 7] =  (self.0[ 1] >>  4)                      as u8;
            s[ 8] =  (self.0[ 1] >> 12)                      as u8;
            s[ 9] =  (self.0[ 1] >> 20)                      as u8;
            s[10] =  (self.0[ 1] >> 28)                      as u8;
            s[11] =  (self.0[ 1] >> 36)                      as u8;
            s[12] =  (self.0[ 1] >> 44)                      as u8;
            s[13] =  (self.0[ 2] >>  0)                      as u8;
            s[14] =  (self.0[ 2] >>  8)                      as u8;
            s[15] =  (self.0[ 2] >> 16)                      as u8;
            s[16] =  (self.0[ 2] >> 24)                      as u8;
            s[17] =  (self.0[ 2] >> 32)                      as u8;
            s[18] =  (self.0[ 2] >> 40)                      as u8;
            s[19] = ((self.0[ 2] >> 48) | (self.0[ 3] << 4)) as u8;
            s[20] =  (self.0[ 3] >>  4)                      as u8;
            s[21] =  (self.0[ 3] >> 12)                      as u8;
            s[22] =  (self.0[ 3] >> 20)                      as u8;
            s[23] =  (self.0[ 3] >> 28)                      as u8;
            s[24] =  (self.0[ 3] >> 36)                      as u8;
            s[25] =  (self.0[ 3] >> 44)                      as u8;
            s[26] =  (self.0[ 4] >>  0)                      as u8;
            s[27] =  (self.0[ 4] >>  8)                      as u8;
            s[28] =  (self.0[ 4] >> 16)                      as u8;
            s[29] =  (self.0[ 4] >> 24)                      as u8;
            s[30] =  (self.0[ 4] >> 32)                      as u8;
            s[31] =  (self.0[ 4] >> 40)                      as u8;
            s
        }

        /// 5×5 widening multiply → [u128; 9]. The exact shape dalek uses.
        pub fn mul_internal(a: &Scalar52, b: &Scalar52) -> [u128; 9] {
            let mut z = [0u128; 9];
            z[0] = m(a.0[0], b.0[0]);
            z[1] = m(a.0[0], b.0[1]) + m(a.0[1], b.0[0]);
            z[2] = m(a.0[0], b.0[2]) + m(a.0[1], b.0[1]) + m(a.0[2], b.0[0]);
            z[3] = m(a.0[0], b.0[3]) + m(a.0[1], b.0[2]) + m(a.0[2], b.0[1]) + m(a.0[3], b.0[0]);
            z[4] = m(a.0[0], b.0[4]) + m(a.0[1], b.0[3]) + m(a.0[2], b.0[2]) + m(a.0[3], b.0[1]) + m(a.0[4], b.0[0]);
            z[5] =                     m(a.0[1], b.0[4]) + m(a.0[2], b.0[3]) + m(a.0[3], b.0[2]) + m(a.0[4], b.0[1]);
            z[6] =                                         m(a.0[2], b.0[4]) + m(a.0[3], b.0[3]) + m(a.0[4], b.0[2]);
            z[7] =                                                             m(a.0[3], b.0[4]) + m(a.0[4], b.0[3]);
            z[8] =                                                                                 m(a.0[4], b.0[4]);
            z
        }

        /// Compute `a - b` (mod L). Verbatim from dalek's u64 scalar.rs.
        ///
        /// The trailing conditional-add (when the borrow underflows the
        /// low 52 bits, add L back) makes this a constant-time signed-vs-
        /// unsigned bridge. Uses a per-function `black_box` via volatile
        /// load to prevent LLVM from inserting `jns` branches.
        pub fn sub(a: &Scalar52, b: &Scalar52) -> Scalar52 {
            fn black_box(value: u64) -> u64 {
                // Same as dalek: ptr::read_volatile to defeat optimization
                unsafe { core::ptr::read_volatile(&value) }
            }
            let mut difference = Scalar52::ZERO;
            let mask = (1u64 << 52) - 1;
            let mut borrow: u64 = 0;
            for i in 0..5 {
                borrow = a.0[i].wrapping_sub(b.0[i] + (borrow >> 63));
                difference.0[i] = borrow & mask;
            }
            let underflow_mask = ((borrow >> 63) ^ 1).wrapping_sub(1);
            let mut carry: u64 = 0;
            for i in 0..5 {
                carry = (carry >> 52) + difference.0[i] + (L.0[i] & black_box(underflow_mask));
                difference.0[i] = carry & mask;
            }
            difference
        }

        /// Same as `montgomery_reduce` but stops before the final
        /// `Scalar52::sub(result, L)` call. Used by slot 85 — preserves
        /// the test point from the v1.46 run where 85 was confirmed to
        /// PASS without sub. Keeping this separate lets us directly
        /// compare "reduce without sub" (slot 85) vs "reduce with sub"
        /// (slot 90) results on each subsequent vast run.
        pub fn montgomery_reduce_no_sub(limbs: &[u128; 9]) -> Scalar52 {
            #[inline(always)]
            fn part1(sum: u128) -> (u128, u64) {
                let p = (sum as u64).wrapping_mul(LFACTOR) & ((1u64 << 52) - 1);
                ((sum + m(p, L.0[0])) >> 52, p)
            }
            #[inline(always)]
            fn part2(sum: u128) -> (u128, u64) {
                let w = (sum as u64) & ((1u64 << 52) - 1);
                (sum >> 52, w)
            }
            let l = &L;
            let (carry, n0) = part1(        limbs[0]);
            let (carry, n1) = part1(carry + limbs[1] + m(n0, l.0[1]));
            let (carry, n2) = part1(carry + limbs[2] + m(n0, l.0[2]) + m(n1, l.0[1]));
            let (carry, n3) = part1(carry + limbs[3]                 + m(n1, l.0[2]) + m(n2, l.0[1]));
            let (carry, n4) = part1(carry + limbs[4] + m(n0, l.0[4])                 + m(n2, l.0[2]) + m(n3, l.0[1]));
            let (carry, r0) = part2(carry + limbs[5]                 + m(n1, l.0[4])                 + m(n3, l.0[2]) + m(n4, l.0[1]));
            let (carry, r1) = part2(carry + limbs[6]                                 + m(n2, l.0[4])                 + m(n4, l.0[2]));
            let (carry, r2) = part2(carry + limbs[7]                                                 + m(n3, l.0[4]));
            let (carry, r3) = part2(carry + limbs[8]                                                                 + m(n4, l.0[4]));
            let         r4 = carry as u64;
            Scalar52([r0, r1, r2, r3, r4])
        }

        /// Compute `limbs / R` (mod L) — exactly dalek's montgomery_reduce
        /// including the final `Scalar52::sub(result, L)` call.
        pub fn montgomery_reduce(limbs: &[u128; 9]) -> Scalar52 {
            #[inline(always)]
            fn part1(sum: u128) -> (u128, u64) {
                let p = (sum as u64).wrapping_mul(LFACTOR) & ((1u64 << 52) - 1);
                ((sum + m(p, L.0[0])) >> 52, p)
            }
            #[inline(always)]
            fn part2(sum: u128) -> (u128, u64) {
                let w = (sum as u64) & ((1u64 << 52) - 1);
                (sum >> 52, w)
            }
            let l = &L;
            let (carry, n0) = part1(        limbs[0]);
            let (carry, n1) = part1(carry + limbs[1] + m(n0, l.0[1]));
            let (carry, n2) = part1(carry + limbs[2] + m(n0, l.0[2]) + m(n1, l.0[1]));
            let (carry, n3) = part1(carry + limbs[3]                 + m(n1, l.0[2]) + m(n2, l.0[1]));
            let (carry, n4) = part1(carry + limbs[4] + m(n0, l.0[4])                 + m(n2, l.0[2]) + m(n3, l.0[1]));
            let (carry, r0) = part2(carry + limbs[5]                 + m(n1, l.0[4])                 + m(n3, l.0[2]) + m(n4, l.0[1]));
            let (carry, r1) = part2(carry + limbs[6]                                 + m(n2, l.0[4])                 + m(n4, l.0[2]));
            let (carry, r2) = part2(carry + limbs[7]                                                 + m(n3, l.0[4]));
            let (carry, r3) = part2(carry + limbs[8]                                                                 + m(n4, l.0[4]));
            let         r4 = carry as u64;
            // The full dalek implementation: result may be >= L, so
            // attempt to subtract L. This was missing from earlier
            // iterations of the port — slot 86 PASSed without it, which
            // told us mul_internal + reduce-without-sub are fine but
            // hid the fact that sub itself might be the bug.
            Scalar52::sub(&Scalar52([r0, r1, r2, r3, r4]), l)
        }
    }
}

const DALEK_ONE_LIMBS: [u64; 5] = [1, 0, 0, 0, 0];

// Slot 84 — Rung A: pure byte→limbs unpack. No arithmetic, no statics
// other than the const masks.
pub fn check_dalek_scalar52_from_bytes() -> u32 {
    let mut bytes = [0u8; 32];
    bytes[0] = 1;
    let bytes = core::hint::black_box(bytes);
    let s = bisect_scalar52::Scalar52::from_bytes(&bytes);
    (s.0 == DALEK_ONE_LIMBS) as u32
}

// Slot 85 — Rung C alone (WITHOUT the final sub call). Calls the
// `montgomery_reduce_no_sub` variant so this slot's result is directly
// comparable to the v1.46 run where it PASSed. Slot 90 calls the same
// pipeline WITH the sub — if 85 PASSes and 90 FAILs, the bug is in sub.
pub fn check_dalek_scalar52_montgomery_reduce_r() -> u32 {
    let r = bisect_scalar52::R;
    let mut widened = [0u128; 9];
    for (i, x) in r.0.iter().enumerate() {
        widened[i] = *x as u128;
    }
    let widened = core::hint::black_box(widened);
    let result = bisect_scalar52::Scalar52::montgomery_reduce_no_sub(&widened);
    (result.0 == DALEK_ONE_LIMBS) as u32
}

// Slot 86 — Rungs B+C: mul_internal + montgomery_reduce_no_sub.
// Keeps "without sub" semantics for direct comparison to the v1.46 run.
pub fn check_dalek_scalar52_mul_internal_then_reduce_one_r() -> u32 {
    use bisect_scalar52::Scalar52;
    const ONE: Scalar52 = Scalar52(DALEK_ONE_LIMBS);
    let one = core::hint::black_box(ONE);
    let r = core::hint::black_box(bisect_scalar52::R);
    let x_r = Scalar52::mul_internal(&one, &r);
    let result = Scalar52::montgomery_reduce_no_sub(&x_r);
    (result.0 == DALEK_ONE_LIMBS) as u32
}

// Slot 87 — Rung D: limbs→bytes pack. Inverse of slot 84.
pub fn check_dalek_scalar52_as_bytes_one() -> u32 {
    use bisect_scalar52::Scalar52;
    const ONE: Scalar52 = Scalar52(DALEK_ONE_LIMBS);
    let one = core::hint::black_box(ONE);
    let bytes = one.as_bytes();
    let mut expected = [0u8; 32];
    expected[0] = 1;
    (bytes == expected) as u32
}

// Slot 88: `Scalar52::sub(R, R) == ZERO`. Pure borrow chain across 5 u64
// limbs, no underflow, no conditional add-L. If this FAILs, the basic
// borrow propagation is broken (Slot 47's overflowing_sub is a SINGLE
// op; this is a 5-limb chain).
pub fn check_dalek_scalar52_sub_no_underflow() -> u32 {
    let r = core::hint::black_box(bisect_scalar52::R);
    let result = bisect_scalar52::Scalar52::sub(&r, &r);
    (result.0 == [0u64; 5]) as u32
}

// Slot 89: `Scalar52::sub(ZERO, ONE)` — triggers the underflow path.
// borrow propagates to the top bit; underflow_mask = all-1s; the
// conditional-add loop adds L back. Mathematically: 0 - 1 mod L = L - 1.
// L - 1 in 5x52-bit limbs:
//   limb[0] = 0x0002631a5cf5d3ec  (L[0] - 1)
//   limb[1..4] = L[1..4] unchanged
pub fn check_dalek_scalar52_sub_with_underflow() -> u32 {
    use bisect_scalar52::Scalar52;
    let zero = core::hint::black_box(Scalar52::ZERO);
    let one = core::hint::black_box(Scalar52(DALEK_ONE_LIMBS));
    let result = Scalar52::sub(&zero, &one);
    let expected: [u64; 5] = [
        0x0002631a5cf5d3ec, // L[0] - 1
        0x000dea2f79cd6581, // L[1]
        0x000000000014def9, // L[2]
        0x0000000000000000, // L[3]
        0x0000100000000000, // L[4]
    ];
    (result.0 == expected) as u32
}

// Slot 90: full montgomery_reduce(widened R) with the final sub call now
// included. Compare to slot 85 (same input, sub-less version): if 85
// PASS and 90 FAIL, the bug is in `Scalar52::sub` specifically — that's
// also what makes the real dalek path (slot 71) FAIL.
pub fn check_dalek_scalar52_montgomery_reduce_with_sub() -> u32 {
    let r = bisect_scalar52::R;
    let mut widened = [0u128; 9];
    for (i, x) in r.0.iter().enumerate() {
        widened[i] = *x as u128;
    }
    let widened = core::hint::black_box(widened);
    let result = bisect_scalar52::Scalar52::montgomery_reduce(&widened);
    (result.0 == DALEK_ONE_LIMBS) as u32
}

// Slot 91: focused Index/IndexMut trait dispatch probe on a tuple
// struct. Mirrors dalek's Scalar52 Index impl shape EXACTLY: tuple
// struct wrapping `[u64; 5]`, Index returns `&u64`, IndexMut returns
// `&mut u64`. If this FAILs, trait dispatch on `[i]` syntax is broken
// on the cuda-oxide alpha-NVPTX backend — explains why dalek (uses
// `a[i]`) fails while our port (uses `a.0[i]`) passes.
pub struct IdxProbe(pub [u64; 5]);

impl core::ops::Index<usize> for IdxProbe {
    type Output = u64;
    fn index(&self, i: usize) -> &u64 {
        &(self.0[i])
    }
}

impl core::ops::IndexMut<usize> for IdxProbe {
    fn index_mut(&mut self, i: usize) -> &mut u64 {
        &mut (self.0[i])
    }
}

pub fn check_index_trait_dispatch() -> u32 {
    let mut p = IdxProbe([0u64; 5]);
    let idx = core::hint::black_box(2usize);
    let val = core::hint::black_box(0xCAFE_BABE_DEAD_BEEF_u64);
    p[idx] = val;
    let read = core::hint::black_box(p[idx]);
    (read == val) as u32
}

// Slot 92: dalek `Scalar::ONE.to_bytes()` direct. Cross-crate access to
// a `pub const Scalar` followed by trivial byte copy (Scalar's internal
// rep IS the bytes; to_bytes just copies them out). No reduce, no math.
// If this FAILs, the bug is at the cross-crate const-access layer.
pub fn check_dalek_scalar_one_to_bytes_direct() -> u32 {
    let s = curve25519_dalek::Scalar::ONE;
    let bytes = s.to_bytes();
    let mut expected = [0u8; 32];
    expected[0] = 1;
    (bytes == expected) as u32
}

// Slot 93: k256 `AffinePoint::GENERATOR.to_encoded_point(true)`. Skips
// the projective→affine conversion that slot 78 includes (no z-coord
// inversion). Tests cross-crate const access for AffinePoint::GENERATOR
// + the encoded_point serialization chain. If 93 PASSes and 78 FAILs,
// the bug in 78 is specifically in `to_affine()` (the field inversion).
pub fn check_k256_affine_generator_encode() -> u32 {
    use k256::AffinePoint;
    use k256::elliptic_curve::sec1::ToEncodedPoint;
    let g = AffinePoint::GENERATOR;
    let encoded = g.to_encoded_point(true);
    let bytes = encoded.as_bytes();
    if bytes.len() != 33 {
        return 0;
    }
    let mut out = [0u8; 33];
    out.copy_from_slice(bytes);
    (out == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 94: subtle::Choice u8 → bool. The most trivial subtle operation.
// Choice is a tuple struct wrapping u8 with field private. From<u8> sets
// it; Into<bool> reads it via debug_assert + comparison.
pub fn check_subtle_choice_u8_into_bool() -> u32 {
    use k256::elliptic_curve::subtle::Choice;
    let c0 = Choice::from(core::hint::black_box(0u8));
    let c1 = Choice::from(core::hint::black_box(1u8));
    let b0: bool = c0.into();
    let b1: bool = c1.into();
    (!b0 && b1) as u32
}

// Slot 95: subtle::ConditionallySelectable on u64. The mechanism k256's
// `AffinePoint::to_encoded_point` uses to pick between the identity
// arm and the from_affine_coordinates arm.
//   conditional_select(&a, &b, Choice(0)) should return a
//   conditional_select(&a, &b, Choice(1)) should return b
// Slot 53/54 tested a HAND-ROLLED mask blend with the same conceptual
// math; this slot tests the actual subtle::ConditionallySelectable trait
// impl which the real code path uses.
pub fn check_subtle_conditional_select_u64() -> u32 {
    use k256::elliptic_curve::subtle::{Choice, ConditionallySelectable};
    let a = core::hint::black_box(0xCAFE_BABE_DEAD_BEEF_u64);
    let b = core::hint::black_box(0x1234_5678_9ABC_DEF0_u64);
    let c0 = Choice::from(core::hint::black_box(0u8));
    let c1 = Choice::from(core::hint::black_box(1u8));
    let r0 = u64::conditional_select(&a, &b, c0);
    let r1 = u64::conditional_select(&a, &b, c1);
    (r0 == a && r1 == b) as u32
}

// Slot 96: `EncodedPoint::from_affine_coordinates(&GX_bytes, &GY_bytes,
// compress=true)` with hardcoded generator-x/y. Bypasses AffinePoint's
// own `to_encoded_point` (which goes through `is_identity`+
// `conditional_select`) and tests just the EncodedPoint construction.
//
// If 96 PASSes and 93 FAILs, the bug is in `is_identity`/`conditional_
// select` (slot 95 should then also FAIL). If 96 FAILs, EncodedPoint
// construction itself is broken.
const SECP256K1_GX_BYTES: [u8; 32] = [
    0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
    0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
    0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
    0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98,
];
const SECP256K1_GY_BYTES: [u8; 32] = [
    0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65,
    0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08, 0xA8,
    0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19,
    0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10, 0xD4, 0xB8,
];

pub fn check_k256_encoded_point_from_affine_coords() -> u32 {
    use k256::EncodedPoint;
    use k256::elliptic_curve::FieldBytes;
    let x: &FieldBytes<k256::Secp256k1> = (&SECP256K1_GX_BYTES).into();
    let y: &FieldBytes<k256::Secp256k1> = (&SECP256K1_GY_BYTES).into();
    let encoded = EncodedPoint::from_affine_coordinates(x, y, true);
    let bytes = encoded.as_bytes();
    if bytes.len() != 33 {
        return 0;
    }
    let mut out = [0u8; 33];
    out.copy_from_slice(bytes);
    (out == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 97: Index/IndexMut trait dispatch with LITERAL const indices.
// Slot 91 used `black_box(idx)` → runtime index, and now PASSes. Dalek's
// Scalar52::from_bytes uses `s[0] = …; s[1] = …; …; s[4] = …` with
// const literal indices. Different IR shape — const indices typically
// fold the trait call into a direct GEP at compile time.
pub fn check_index_trait_const_indices() -> u32 {
    let mut p = IdxProbe([0u64; 5]);
    p[0] = core::hint::black_box(0x1111_1111_1111_1111_u64);
    p[1] = core::hint::black_box(0x2222_2222_2222_2222_u64);
    p[2] = core::hint::black_box(0x3333_3333_3333_3333_u64);
    p[3] = core::hint::black_box(0x4444_4444_4444_4444_u64);
    p[4] = core::hint::black_box(0x5555_5555_5555_5555_u64);
    let r0 = p[0];
    let r1 = p[1];
    let r2 = p[2];
    let r3 = p[3];
    let r4 = p[4];
    (r0 == 0x1111_1111_1111_1111
        && r1 == 0x2222_2222_2222_2222
        && r2 == 0x3333_3333_3333_3333
        && r3 == 0x4444_4444_4444_4444
        && r4 == 0x5555_5555_5555_5555) as u32
}

// Slot 98: `GenericArray<u8, U33>` basic index. GenericArray doesn't have
// a custom Index impl; it Derefs to `[T]` via:
//   `unsafe { slice::from_raw_parts(self as *const Self as *const T, N::USIZE) }`
// If that raw-ptr-cast Deref miscompiles, every GenericArray op breaks.
// k256::EncodedPoint stores its bytes in a `GenericArray<u8, EncodedSize>`.
pub fn check_generic_array_basic_index() -> u32 {
    use k256::elliptic_curve::generic_array::GenericArray;
    use k256::elliptic_curve::generic_array::typenum::U33;
    let mut ga: GenericArray<u8, U33> = GenericArray::default();
    let i0 = core::hint::black_box(0usize);
    let i32 = core::hint::black_box(32usize);
    ga[i0] = 0xAA;
    ga[i32] = 0xBB;
    let v0 = ga[i0];
    let v32 = ga[i32];
    (v0 == 0xAA && v32 == 0xBB) as u32
}

// Slot 99: `GenericArray<u8, U33>` populated via `copy_from_slice` from a
// regular byte array. This is exactly what EncodedPoint::from_affine_
// coordinates does:
//   bytes[1..33].copy_from_slice(x);
// If this FAILs, the slice-copy-into-GenericArray-slice is the bug.
pub fn check_generic_array_copy_from_slice() -> u32 {
    use k256::elliptic_curve::generic_array::GenericArray;
    use k256::elliptic_curve::generic_array::typenum::U33;
    let src: [u8; 32] = SECP256K1_GX_BYTES;
    let mut ga: GenericArray<u8, U33> = GenericArray::default();
    ga[0] = 0x02;
    ga[1..33].copy_from_slice(&src);
    // Compare against the known compressed-generator encoding.
    let mut got = [0u8; 33];
    got.copy_from_slice(&ga[..]);
    (got == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 100: local re-impl of sec1's `from_affine_coordinates` body using
// raw `[u8; 33]` instead of `GenericArray<u8, U33>`. Same algorithm:
//   tag = 0x02/0x03 based on y[31]&1
//   bytes[0] = tag
//   bytes[1..33] = x
// If 100 PASSes and 96 still FAILs, the bug is in sec1's
// GenericArray-typed parameter handling, not the algorithm.
pub fn check_from_affine_coords_replica() -> u32 {
    let x_bytes = &SECP256K1_GX_BYTES;
    let y_bytes = &SECP256K1_GY_BYTES;
    // Compute tag: even y → 0x02, odd y → 0x03
    let last_y = core::hint::black_box(y_bytes[31]);
    let tag: u8 = if last_y & 1 == 1 { 0x03 } else { 0x02 };
    let mut bytes = [0u8; 33];
    bytes[0] = tag;
    bytes[1..33].copy_from_slice(x_bytes);
    (bytes == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 101: probes the exact `y.as_slice().last()` shape inside
// `Tag::compress_y`. Pass a `&GenericArray<u8, U32>` to a function, do
// `as_slice().last()` inside. Slot 99 tested write-side copy; this
// tests read-side slice access via Deref then `.last()`.
#[inline(never)]
fn last_via_as_slice(ga: &k256::elliptic_curve::generic_array::GenericArray<u8, k256::elliptic_curve::generic_array::typenum::U32>) -> u8 {
    *ga.as_slice().last().expect("non-empty")
}

pub fn check_generic_array_as_slice_last() -> u32 {
    use k256::elliptic_curve::generic_array::GenericArray;
    use k256::elliptic_curve::generic_array::typenum::U32;
    let ga: &GenericArray<u8, U32> = (&SECP256K1_GY_BYTES).into();
    let last = last_via_as_slice(ga);
    (last == 0xB8) as u32   // SECP256K1_GY_BYTES[31]
}

// Slot 102: dalek `Scalar::from_bytes_mod_order([0; 32])` should
// round-trip to all-zeros (the canonical encoding of 0). Mirror of slot
// 71 with a different input value. If 102 PASS but 71 FAIL, the bug is
// input-dependent (only non-zero scalars). If 102 FAIL too, the bug is
// general to any Scalar::from_bytes_mod_order call.
pub fn check_dalek_scalar_round_trip_zero() -> u32 {
    let input = [0u8; 32];
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(core::hint::black_box(input));
    let bytes = scalar.to_bytes();
    (bytes == input) as u32
}

// Slot 103: `Scalar::from_bytes_mod_order_wide(&[0; 64])` — uses a
// different reduction entry point than slot 71/102. Internally it calls
// `Scalar52::from_bytes_wide` + `montgomery_mul(R)` / `montgomery_mul(RR)`
// composition, NOT `Scalar::reduce`. For all-zero input the result is
// canonically 0.
pub fn check_dalek_scalar_from_bytes_wide_zero() -> u32 {
    let input = [0u8; 64];
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order_wide(
        &core::hint::black_box(input),
    );
    let bytes = scalar.to_bytes();
    (bytes == [0u8; 32]) as u32
}

// Slot 104: `(&[u8; 32]).into() → &FieldBytes<Secp256k1>` then read first
// and last bytes. Tests the `From<&[u8; N]> for &GenericArray<u8, N>`
// conversion (the only GA-related path slot 98/99 didn't cover — they
// constructed via `GenericArray::default()` instead).
pub fn check_field_bytes_into_conversion() -> u32 {
    use k256::elliptic_curve::FieldBytes;
    let arr: [u8; 32] = SECP256K1_GX_BYTES;
    let arr = core::hint::black_box(arr);
    let ga: &FieldBytes<k256::Secp256k1> = (&arr).into();
    let first = ga[0];
    let last = ga[31];
    (first == 0x79 && last == 0x98) as u32
}

// Slot 105: `base58_encode_32` with minimum non-zero input: 31 leading
// zero bytes + 1 byte of value 1. Forces `limb_count == 1` after the
// outer loop (vs slot 41 which has higher limb_count). Expected output
// is 31 '1's followed by '2' = 32 chars.
pub fn check_base58_min_nonzero() -> u32 {
    let mut input = [0u8; 32];
    input[31] = 1;
    let mut out = [0u8; 64];
    let n = base58_encode_32(&input, &mut out);
    let expected: &[u8] = b"11111111111111111111111111111112";
    if n != expected.len() {
        return 0;
    }
    let mut i = 0;
    while i < n {
        if out[i] != expected[i] {
            return 0;
        }
        i += 1;
    }
    1
}

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

pub fn check_named_field_struct_return() -> u32 {
    let input = core::hint::black_box([0u8; 32]);
    let out = make_wrap_named(input);
    let mut expected = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        expected[i] = 1;
        i += 1;
    }
    (out.bytes == expected) as u32
}

// Slot 107: hand-rolled base58 of [0; 31] + [0x01] without the `seq!`
// macro. base58_encode_32 uses `seq!(I in 0..8 { ... })` which proc-macro-
// unrolls the outer loop 8 times. This replica uses a plain `for I in
// 0..8` instead, with otherwise-identical body. Same algorithm, same
// constants, just no seq! expansion.
//
// If 107 PASS but 105 FAIL, the bug is in something specific to seq!'s
// expansion (function size, code layout, etc).
// Slot 108: `<[u8]>::reverse()` on a partial sub-slice. The only
// operation in `base58_encode_32` that slot 107 (which PASSed) hand-
// rolls — slot 107 uses manual swap pairs, while the original calls
// `output[..result_len].reverse()`. If 108 FAILs, that's the Bug-41
// minimal repro.
pub fn check_slice_reverse_partial() -> u32 {
    let mut arr = [0u8; 64];
    // Populate a non-trivial prefix with a recognizable pattern
    arr[0] = 0x11;
    arr[1] = 0x22;
    arr[2] = 0x33;
    arr[3] = 0x44;
    arr[4] = 0x55;
    let result_len = core::hint::black_box(5usize);
    arr[..result_len].reverse();
    // Expected after reverse: [0x55, 0x44, 0x33, 0x22, 0x11, 0, 0, ...]
    let mut expected = [0u8; 64];
    expected[0] = 0x55;
    expected[1] = 0x44;
    expected[2] = 0x33;
    expected[3] = 0x22;
    expected[4] = 0x11;
    (arr == expected) as u32
}

// Slot 109: `Scalar::from_bytes_mod_order([0; 32]) == Scalar::ZERO`
// using dalek's `PartialEq` (which uses constant-time equality
// internally) instead of comparing the bytes output of `to_bytes`.
// Disambiguates: is the Scalar VALUE correct, or is `to_bytes` broken?
//   If 109 PASS but 102 FAIL → bug is specifically in `to_bytes`.
//   If 109 FAIL → the Scalar value from `from_bytes_mod_order` is wrong.
pub fn check_dalek_scalar_eq_zero() -> u32 {
    use curve25519_dalek::Scalar;
    let input = core::hint::black_box([0u8; 32]);
    let s = Scalar::from_bytes_mod_order(input);
    let zero = Scalar::ZERO;
    (s == zero) as u32
}

// Slot 110: `dst_ga.copy_from_slice(src_ga)` where source IS a
// `&GenericArray<u8, U32>` (not `&[u8; 32]`). Slot 99 already covered
// `&[u8; 32]` source. The function `EncodedPoint::from_affine_coordinates`
// uses `bytes[1..33].copy_from_slice(x)` where `x: &GenericArray`, so
// the source-side Deref→slice conversion happens implicitly.
pub fn check_generic_array_copy_from_ga_source() -> u32 {
    use k256::elliptic_curve::generic_array::GenericArray;
    use k256::elliptic_curve::generic_array::typenum::{U32, U33};
    let src_arr = SECP256K1_GX_BYTES;
    let src: &GenericArray<u8, U32> = (&src_arr).into();
    let mut dst: GenericArray<u8, U33> = GenericArray::default();
    dst[0] = 0x02;
    dst[1..33].copy_from_slice(src);
    let mut got = [0u8; 33];
    got.copy_from_slice(&dst);
    (got == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 111: `Scalar::ZERO == Scalar::ZERO`. Pure const-vs-const
// PartialEq, no function call producing a Scalar. If FAIL, dalek's
// PartialEq impl itself is broken; if PASS, slot 109's FAIL is
// genuinely from from_bytes_mod_order returning a non-zero value.
pub fn check_dalek_zero_eq_zero() -> u32 {
    use curve25519_dalek::Scalar;
    let a = core::hint::black_box(Scalar::ZERO);
    let b = core::hint::black_box(Scalar::ZERO);
    (a == b) as u32
}

// Slot 112: `Scalar::from_canonical_bytes([0; 32]).unwrap() == ZERO`.
// `from_canonical_bytes` does NOT call `reduce()` — it just validates
// the bytes are < ℓ and wraps. For [0; 32], 0 < ℓ so it returns
// `CtOption::Some(Scalar { bytes: [0; 32] })`. If 112 PASSes but slot
// 109 FAILs, the bug is in `reduce()` specifically (not the wider
// Scalar construction).
pub fn check_dalek_from_canonical_zero() -> u32 {
    use curve25519_dalek::Scalar;
    let opt = Scalar::from_canonical_bytes(core::hint::black_box([0u8; 32]));
    let s_opt: Option<Scalar> = opt.into();
    let s = match s_opt {
        Some(s) => s,
        None => return 0,
    };
    let zero = Scalar::ZERO;
    (s == zero) as u32
}

pub fn check_base58_handrolled_no_seq() -> u32 {
    const D: u64 = 58_u64.pow(5);
    const DIVISORS: [u64; 5] = [1, 58, 3364, 195112, 11316496];
    const BASE58_ALPHABET: &[u8; 58] =
        b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    let mut input = [0u8; 32];
    input[31] = 1;
    let input = core::hint::black_box(input);

    // num_leading_zeros
    let mut num_leading_zeros: usize = 0;
    let mut i = 0;
    while i < 32 {
        if input[i] == 0 {
            num_leading_zeros += 1;
        } else {
            break;
        }
        i += 1;
    }

    // chunks
    let mut chunks = [0u32; 8];
    let mut c = 0;
    while c < 8 {
        chunks[c] = u32::from_be_bytes([
            input[c * 4],
            input[c * 4 + 1],
            input[c * 4 + 2],
            input[c * 4 + 3],
        ]);
        c += 1;
    }

    // outer loop — same body as base58_encode_32 but plain Rust, no seq!
    let mut limbs = [0u32; 10];
    let mut limb_count: usize = 0;
    let mut k = 0;
    while k < 8 {
        let chunk = chunks[k];
        let carry = chunk as u64;
        let mut remaining_carry = carry;

        let mut j = 0;
        while j < limb_count {
            remaining_carry += (limbs[j] as u64) << 32;
            limbs[j] = (remaining_carry % D) as u32;
            remaining_carry /= D;
            j += 1;
        }

        if remaining_carry > 0 && limb_count < 10 {
            limbs[limb_count] = (remaining_carry % D) as u32;
            remaining_carry /= D;
            limb_count += 1;
            if remaining_carry > 0 && limb_count < 10 {
                limbs[limb_count] = remaining_carry as u32;
                limb_count += 1;
            }
        }
        k += 1;
    }

    // digit extraction
    let mut output = [0u8; 64];
    let mut idx = limb_count;
    while idx > 0 {
        idx -= 1;
        let limb_value = limbs[idx] as u64;
        let output_offset = idx * 5;
        let mut di = 0;
        while di < 5 {
            output[output_offset + di] = ((limb_value / DIVISORS[di]) % 58) as u8;
            di += 1;
        }
    }

    let mut result_len = limb_count * 5;
    while result_len > 0 && output[result_len - 1] == 0 {
        result_len -= 1;
    }

    let mut z = 0;
    while z < num_leading_zeros {
        output[result_len] = 0;
        result_len += 1;
        z += 1;
    }

    let mut a = 0;
    while a < result_len {
        output[a] = BASE58_ALPHABET[output[a] as usize];
        a += 1;
    }

    // reverse output[..result_len]
    let mut lo = 0;
    let mut hi = result_len;
    while lo + 1 < hi {
        hi -= 1;
        let tmp = output[lo];
        output[lo] = output[hi];
        output[hi] = tmp;
        lo += 1;
    }

    let expected: &[u8] = b"11111111111111111111111111111112";
    if result_len != expected.len() {
        return 0;
    }
    let mut x = 0;
    while x < result_len {
        if output[x] != expected[x] {
            return 0;
        }
        x += 1;
    }
    1
}

pub fn run_self_test(results: &mut [u32]) {
    results[0] = check_primitive_xoroshiro();
    results[1] = check_primitive_sha512();
    results[2] = check_primitive_ed25519();
    results[3] = check_primitive_base58();
    results[4] = check_primitive_secp256k1_compressed();
    results[5] = check_primitive_secp256k1_uncompressed();
    results[6] = check_primitive_keccak256();
    results[7] = check_primitive_ripemd160();
    results[8] = check_primitive_sha256_32();
    results[9] = check_primitive_sha256_variable();
    results[10] = check_solana_priv();
    results[11] = check_solana_pub();
    results[12] = check_solana_encoded();
    results[13] = check_ethereum_priv();
    results[14] = check_ethereum_pub();
    results[15] = check_ethereum_address();
    results[16] = check_bitcoin_priv();
    results[17] = check_bitcoin_pub();
    results[18] = check_bitcoin_pkh();
    results[19] = check_bitcoin_encoded();
    results[20] = check_bitcoin_matches();
    results[21] = check_wif_compressed_mainnet();
    results[22] = check_wif_uncompressed_mainnet();
    results[23] = check_wif_compressed_testnet();
    results[24] = check_wif_uncompressed_testnet();
    results[25] = check_shallenge_hash();
    results[26] = check_shallenge_nonce_len();
    results[27] = check_shallenge_is_better();
    results[28] = check_compare_hashes_lt();
    results[29] = check_compare_hashes_gt();
    results[30] = check_compare_hashes_eq();
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
    results[42] = check_base58_var_len_leading_zero();
    results[43] = check_base58_all_zeros();
    results[44] = check_xoroshiro_base64_nonce();
    results[45] = check_bech32_p2wpkh();
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
    results[73] = check_k256_secret_from_bytes_one();
    results[74] = check_k256_derive_scalar_one();
    results[75] = check_k256_derive_scalar_two();
    results[76] = check_static_u64_array_lookup();
    results[77] = check_static_struct_wrapped_u64_lookup();
    results[78] = check_k256_encode_generator();
    results[79] = check_k256_double_generator();
    results[80] = check_k256_scalar_one_round_trip();
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
    results[93] = check_k256_affine_generator_encode();
    results[94] = check_subtle_choice_u8_into_bool();
    results[95] = check_subtle_conditional_select_u64();
    results[96] = check_k256_encoded_point_from_affine_coords();
    results[97] = check_index_trait_const_indices();
    results[98] = check_generic_array_basic_index();
    results[99] = check_generic_array_copy_from_slice();
    results[100] = check_from_affine_coords_replica();
    results[101] = check_generic_array_as_slice_last();
    results[102] = check_dalek_scalar_round_trip_zero();
    results[103] = check_dalek_scalar_from_bytes_wide_zero();
    results[104] = check_field_bytes_into_conversion();
    results[105] = check_base58_min_nonzero();
    results[106] = check_named_field_struct_return();
    results[107] = check_base58_handrolled_no_seq();
    results[108] = check_slice_reverse_partial();
    results[109] = check_dalek_scalar_eq_zero();
    results[110] = check_generic_array_copy_from_ga_source();
    results[111] = check_dalek_zero_eq_zero();
    results[112] = check_dalek_from_canonical_zero();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn all_self_test_checks_pass_on_cpu() {
        let mut results = [0u32; SELF_TEST_NUM_CHECKS];
        run_self_test(&mut results);
        for (i, &r) in results.iter().enumerate() {
            assert_eq!(r, 1, "self-test check {} ({}) failed", i, SELF_TEST_LABELS[i]);
        }
    }
}
