# Known GPU self-test failures

Tracks failures from `compute-sanitizer --tool memcheck ./vanity-miner self-test`
against the current cuda-oxide alpha-NVPTX compiler. Updated after every vast run.

Self-contained: includes both our self-test layer and the downstream-crate
Rust the failing call paths execute, so this doc is enough context to hand
to the compiler agent.

## Run history

| Date       | Suite | FAIL | cuda-oxide | Notes |
|------------|-------|------|------------|-------|
| 2026-05-13 | 76    | 25   | v1.43.0    | Bug C still failing. First Bug A hypothesis. |
| 2026-05-13 | 81    | 27   | v1.46.0    | Bug C fixed. Static-array hypothesis falsified. |
| 2026-05-13 | 88    | 27   | v1.46.0    | Ladder slots 84-87 PASS — verbatim Scalar52 port works inside `logic`. |
| 2026-05-13 | 91    | 27   | v1.46.0    | Sub probes 88-90 PASS. |
| 2026-05-13 | 94    | 29   | v1.46.0    | Slot 91 FAIL (later flipped). |
| 2026-05-13 | 97    | 29   | v1.46.0    | Slot 91 PASS (Bug E lost), 96 FAIL (Bug-96 narrowed). |
| pending    | 100   | ?    | next       | +3 probes (97-99). |

---

## Currently open bugs

Three open compiler bugs. Names use the direct-hit probe's slot number —
the name points to the minimal repro.

### Bug-71 — dalek `Scalar::from_bytes_mod_order` chain

**Status.** Direct repro slot 71 FAILs every run. No isolated minimal
repro yet — slots 81-90 (ladder of every intermediate step of dalek's
`Scalar::reduce`, ported verbatim into `logic`) all PASS.

**Slot 71 (smoking gun):**
```rust
pub fn check_dalek_scalar_round_trip_one() -> u32 {
    let mut input = [0u8; 32];
    input[0] = 1;
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(input);
    let bytes = scalar.to_bytes();
    (bytes == input) as u32   // For scalar=1, should round-trip identity
}
```

**Ruled out** (all PASS):
- `u128 >> 52` immediate shift (slot 81)
- `u128 + u128` carry chains (slot 65)
- `&'static [u64; N]` reads, raw or newtype-wrapped (slots 76/77/82)
- `Scalar52::from_bytes`, `mul_internal`, `montgomery_reduce_no_sub`, `sub`, `as_bytes` (verbatim ports — slots 84-90)
- Simple `Index<usize>`/`IndexMut<usize>` trait dispatch with `black_box(idx)` (slot 91 PASS)

**Live hypothesis.** Dalek's `Scalar52` uses `s[0] = …; s[1] = …` with
LITERAL const indices via custom `Index<usize>` trait. Slot 91 used
`black_box(idx)` (runtime) — different LLVM IR shape. Const indices may
fold the trait call into a direct GEP that miscompiles separately.

**Probe in flight:** Slot 97 — `IdxProbe` with 5 literal const indices
in a row (mirrors dalek `Scalar52::from_bytes`'s `s[0]..=s[4]` pattern).

**Failures owned:** 5 (slots 2, 11, 12, 71, 72) + partial 3/12.

---

### Bug-41 — base58 encoding with non-zero input

**Status.** Direct repro slots 41/42 FAIL every run. Slot 43 (all-zero
input) PASSes because the all-zero path skips the digit-extraction loop.
Every building-block primitive PASSes in isolation; bug is in an
interaction we haven't isolated.

**Slot 41 (smoking gun):**
```rust
pub fn check_base58_var_len() -> u32 {
    let input: [u8; 32] = [/* non-zero 32 bytes */];
    let mut out = [0u8; 64];
    let n = base58_encode_32(&input, &mut out);
    n == BASE58_VAR_EXPECTED.len()
        && &out[..n] == BASE58_VAR_EXPECTED
}
```

**Ruled out** (all PASS in isolation):
- Reverse range iter `(0..n).rev()` (slot 83)
- Phase A inner-mutate loop body (slot 69)
- Phase B grow-tail (slot 67)
- `dividend = carry + (limb << 32); /D; %D` shape (slot 66)
- `&'static [u64; 5]` DIVISORS reads (slots 76/77)
- `BASE58_ALPHABET` lookup (slots 60-62)

**Next move.** No probe queued. Strategy: wait on Bug-71 and Bug-96 to
settle — one of those fixes may cascade here. If not, write a minimized
`base58_encode_8` (just 2 outer iterations) to find the threshold where
the bug appears.

**Failures owned:** 2 direct (41, 42) + partial cascade in 3, 12, 19, 21-24.

---

### Bug-96 — `sec1::EncodedPoint::from_affine_coordinates`

**Status.** Narrowed by slot 96 FAIL (slots 94/95 PASS — `subtle::Choice`
and `ConditionallySelectable` are fine). The bug is inside
`from_affine_coordinates` or its `GenericArray` dependency.

**Slot 96 (smoking gun):**
```rust
pub fn check_k256_encoded_point_from_affine_coords() -> u32 {
    let x: &FieldBytes<Secp256k1> = (&SECP256K1_GX_BYTES).into();
    let y: &FieldBytes<Secp256k1> = (&SECP256K1_GY_BYTES).into();
    let encoded = EncodedPoint::from_affine_coordinates(x, y, true);
    // Compare 33 bytes to SECP256K1_GENERATOR_COMPRESSED
}
```

**Downstream** ([sec1-0.7.3/src/point.rs:121](sec1/point.rs)):
```rust
pub fn from_affine_coordinates(x: &GA<u8,N>, y: &GA<u8,N>, compress: bool) -> Self {
    let tag = if compress { Tag::compress_y(y.as_slice()) } else { Tag::Uncompressed };
    let mut bytes = GenericArray::default();
    bytes[0] = tag.into();                                // IndexMut on GA
    bytes[1..(Size::to_usize() + 1)].copy_from_slice(x);  // slice copy
    if !compress { bytes[(Size::to_usize() + 1)..].copy_from_slice(y); }
    Self { bytes }
}
```

**Suspect.** `GenericArray<T, N>` has no custom `Index` impl — it
`Deref`s to `[T]` via `unsafe { slice::from_raw_parts(self as *const
Self as *const T, N::USIZE) }`. If that raw-ptr-cast Deref is
miscompiled, every `bytes[i]` and `bytes[a..b]` on a GenericArray is
wrong.

**Probes in flight:**
- Slot 98 — `GenericArray<u8, U33>::default()` then `ga[0]/ga[32]` write+read.
- Slot 99 — `GenericArray<u8, U33>` via `ga[1..33].copy_from_slice(&src)`.

Decoding: 98 FAIL → GA Deref broken (explains everything). 99 FAIL → GA
slice copy broken. Both PASS → bug is in `Tag::compress_y(y.as_slice())`
or elsewhere — need a new probe round.

**Failures owned:** 22 (slots 4, 5, 13-24, 74, 75, 78, 79, 80, 93, 96).

---

## Per-failure inventory (29 failures, v1.46.0)

| #  | Slot name                              | Bug             | Why |
|----|----------------------------------------|-----------------|-----|
|  2 | ed25519 derive                         | Bug-71          | cascade of 71 |
|  3 | base58 encode pub                      | Bug-71 + Bug-41 | base58 of dalek output |
|  4 | secp256k1 compressed                   | Bug-96          | k256 derive ends in `to_encoded_point` |
|  5 | secp256k1 uncompressed                 | Bug-96          | cascade of 4 |
| 11 | solana pub                             | Bug-71          | cascade of 2 |
| 12 | solana encoded                         | Bug-71 + Bug-41 | cascade of 2 + base58 |
| 13 | ethereum priv                          | Bug-96          | cascade of 4 |
| 14 | ethereum pub                           | Bug-96          | cascade of 4 |
| 15 | ethereum address                       | Bug-96          | cascade of 4 |
| 16 | bitcoin priv                           | Bug-96          | cascade of 4 |
| 17 | bitcoin pub                            | Bug-96          | cascade of 4 |
| 18 | bitcoin pkh                            | Bug-96          | cascade of 4 |
| 19 | bitcoin encoded                        | Bug-96 + Bug-41 | cascade of 4 + base58 |
| 20 | bitcoin matches                        | Bug-96          | cascade of 4 |
| 21 | wif compressed mainnet                 | Bug-96 + Bug-41 | cascade of 4 + base58 |
| 22 | wif uncompressed mainnet               | Bug-96 + Bug-41 | cascade of 4 + base58 |
| 23 | wif compressed testnet                 | Bug-96 + Bug-41 | cascade of 4 + base58 |
| 24 | wif uncompressed testnet               | Bug-96 + Bug-41 | cascade of 4 + base58 |
| 41 | base58 var-len                         | Bug-41          | **direct hit** |
| 42 | base58 var-len leading-zero            | Bug-41          | variant of 41 |
| 71 | dalek scalar from-bytes round-trip     | Bug-71          | **direct hit** |
| 72 | dalek mul_base scalar=1                | Bug-71          | depends on 71 |
| 74 | k256 derive scalar=1                   | Bug-96          | calls `to_encoded_point` |
| 75 | k256 derive scalar=2                   | Bug-96          | calls `to_encoded_point` |
| 78 | k256 encode generator (no mul)         | Bug-96          | direct path to `to_encoded_point` |
| 79 | k256 double generator + encode         | Bug-96          | 78 + doubling |
| 80 | k256 Scalar::ONE round-trip            | Bug-96 (likely) | `Scalar::to_repr` returns `GenericArray<u8, U32>` |
| 93 | k256 AffinePoint encode                | Bug-96          | direct k256 path |
| 96 | EncodedPoint::from_affine_coordinates  | Bug-96          | **direct hit** |

29 rows = 29 failing slots; every failure has a primary suspect.

---

## Active probes awaiting next run

| Slot | Targets | Probe |
|---|---|---|
| 97 | Bug-71 | `IdxProbe` with 5 literal-const-index writes/reads |
| 98 | Bug-96 | `GenericArray<u8, U33>` basic write/read at fixed indices |
| 99 | Bug-96 | `GenericArray<u8, U33>` populated via `copy_from_slice` |

---

## Not bugs (clarifications)

- **Slot 43 (base58 all-zeros) PASS**: all-zero input → `limb_count` stays
  0 → digit-extraction loop with `DIVISORS[i]` never runs.
- **Slot 73 (`SecretKey::from_bytes(1)`) PASS** but 74/80 FAIL: 73 only
  checks validation ok-ness; doesn't inspect the inner scalar.
- **Slots 76/77 PASS** despite 41/42 FAIL: `&'static [u64; N]` reads
  work. base58 bug is not in `DIVISORS` reads.
- **Slot 65 PASS now** (was FAIL): Bug-C fixed in v1.46.

---

## History — fixed

### Bug-C (FIXED in v1.46) — u128/i128 carry chain
Slot 65 was the direct hit. Standalone upstream repro lived at
`crates/rustc-codegen-cuda/examples/i128_add_carry_chain` (commit
b6fe7ff in cuda-oxide). Fixed in v1.46.0.

---

## History — falsified hypotheses

Each was a live hypothesis at some point; new probe data killed it. Kept
briefly so we don't re-test the same shape.

| Hypothesis | Falsified by | What we know now |
|---|---|---|
| `&'static` multi-byte-element array reads broken (Bug A v1) | slots 76/77 PASS | `[u64; N]` static reads work, raw and depth-1 newtype-wrapped |
| `u128 >> 52` immediate right shift broken (Bug D1) | slot 81 PASS | u128 immediate shifts work |
| Depth-4 `&'static` newtype nesting broken (Bug D2) | slot 82 PASS | 4-level newtype field projections work |
| `(0..n).rev()` reverse range iterator broken (Bug D3) | slot 83 PASS | reverse range writes work |
| Simple `Index<usize>` trait dispatch broken (Bug E v1) | slot 91 FLIPPED to PASS | runtime-index Index works; const-index under test (slot 97) |
| Some intermediate of dalek's Scalar reduce path miscompiles | slots 84-90 all PASS | each step is correct in isolation; bug only emerges in dalek's exact compilation |
| `Scalar52::sub` borrow-chain / volatile `black_box` broken | slots 88-90 PASS | sub works correctly |
| k256 cascade is `Lazy<>` / once_cell init failure | (not tested directly — folded into Bug-96 narrowing) | superseded by slot-96 narrowing |

---

## Workflow

For each new vast run:

1. Update the run-history table with date / counts / cuda-oxide rev.
2. If a slot moves PASS↔FAIL, update its row in the per-failure inventory.
3. If a probe falsifies a hypothesis, move it to "History — falsified"
   with a one-row entry.
4. If a bug is fixed, move its section to "History — fixed" with the
   version tag.
5. Add new probes under the relevant "Currently open" bug section
   (and in "Active probes awaiting next run").
