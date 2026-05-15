use cuda_std::prelude::*;

/// Self-test plumbing probe: writes `results[0] = 1` and nothing else.
/// Launched before the per-slot kernels to isolate "PTX load + launch +
/// DtoH copy" health from any logic body. If this fails, the bug isn't
/// in `logic::*` — it's in the kernel-load / launch path.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_stub(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[0] = 1;
}

// Per-slot self-test kernels. Each writes its single `1`/`0` result into
// `results[slot]`. Launched as 1×1 grid×block — these are deterministic
// single-point KAT checks. Splitting per slot keeps each kernel's stack
// footprint to one subsystem so an illegal-address fault localizes
// (and slots that ran before the fault still produce reliable values
// before the context goes sticky-errored).
//
// Slot ordering matches `logic::SELF_TEST_LABELS`. See `logic::self_test`
// for the underlying `check_*` bodies (CPU mode calls them too via
// `logic::run_self_test`).
// Slots 0-3: solana per-primitive bisect. If a fault localizes here,
// we know which of xoroshiro / sha512 / ed25519 / base58 is the
// culprit before the composed `solana priv` kernel inlines all four.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_primitive_xoroshiro(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[0] = logic::check_primitive_xoroshiro();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_primitive_sha512(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[1] = logic::check_primitive_sha512();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_primitive_ed25519(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[2] = logic::check_primitive_ed25519();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_primitive_base58(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[3] = logic::check_primitive_base58();
}

// Slots 4-9: non-solana primitive bisect — secp256k1 (compressed +
// uncompressed), keccak256, ripemd160, sha256 (fixed-32 + variable).
// These are the primitives the bitcoin / ethereum / shallenge / WIF
// pipelines compose; isolating each one means a fault here pinpoints
// the broken primitive before the composed kernels (slots 10+) would
// have inlined it.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_primitive_secp256k1_compressed(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[4] = logic::check_primitive_secp256k1_compressed();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_primitive_secp256k1_uncompressed(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[5] = logic::check_primitive_secp256k1_uncompressed();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_primitive_keccak256(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[6] = logic::check_primitive_keccak256();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_primitive_ripemd160(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[7] = logic::check_primitive_ripemd160();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_primitive_sha256_32(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[8] = logic::check_primitive_sha256_32();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_primitive_sha256_variable(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[9] = logic::check_primitive_sha256_variable();
}

// Slots 10-30: composed-subsystem KAT checks.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_solana_priv(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[10] = logic::check_solana_priv();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_solana_pub(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[11] = logic::check_solana_pub();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_solana_encoded(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[12] = logic::check_solana_encoded();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_ethereum_priv(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[13] = logic::check_ethereum_priv();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_ethereum_pub(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[14] = logic::check_ethereum_pub();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_ethereum_address(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[15] = logic::check_ethereum_address();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_bitcoin_priv(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[16] = logic::check_bitcoin_priv();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_bitcoin_pub(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[17] = logic::check_bitcoin_pub();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_bitcoin_pkh(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[18] = logic::check_bitcoin_pkh();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_bitcoin_encoded(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[19] = logic::check_bitcoin_encoded();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_bitcoin_matches(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[20] = logic::check_bitcoin_matches();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_wif_compressed_mainnet(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[21] = logic::check_wif_compressed_mainnet();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_wif_uncompressed_mainnet(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[22] = logic::check_wif_uncompressed_mainnet();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_wif_compressed_testnet(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[23] = logic::check_wif_compressed_testnet();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_wif_uncompressed_testnet(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[24] = logic::check_wif_uncompressed_testnet();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_shallenge_hash(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[25] = logic::check_shallenge_hash();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_shallenge_nonce_len(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[26] = logic::check_shallenge_nonce_len();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_shallenge_is_better(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[27] = logic::check_shallenge_is_better();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_compare_hashes_lt(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[28] = logic::check_compare_hashes_lt();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_compare_hashes_gt(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[29] = logic::check_compare_hashes_gt();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_compare_hashes_eq(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[30] = logic::check_compare_hashes_eq();
}

// Slots 31-40: raw-arithmetic bisect — one PTX op per kernel. Each
// baselines against a host-rustc `const`-evaluated expected value, so a
// PASS on CPU + FAIL on GPU isolates the codegen bug to that exact op.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_u32_div_var(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[31] = logic::check_arith_u32_div_var();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_u32_div_const(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[32] = logic::check_arith_u32_div_const();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_u64_div_var(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[33] = logic::check_arith_u64_div_var();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_u64_div_const(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[34] = logic::check_arith_u64_div_const();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_u32_rem_var(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[35] = logic::check_arith_u32_rem_var();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_u64_rem_var(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[36] = logic::check_arith_u64_rem_var();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_u32_mul_lo(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[37] = logic::check_arith_u32_mul_lo();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_u64_mul_lo(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[38] = logic::check_arith_u64_mul_lo();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_u64_mul_hi(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[39] = logic::check_arith_u64_mul_hi();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_u128_mul(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[40] = logic::check_arith_u128_mul();
}

// Slots 41-45: composed-primitive sub-bisects.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_base58_var_len(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[41] = logic::check_base58_var_len();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_base58_var_len_leading_zero(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[42] = logic::check_base58_var_len_leading_zero();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_base58_all_zeros(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[43] = logic::check_base58_all_zeros();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_xoroshiro_base64_nonce(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[44] = logic::check_xoroshiro_base64_nonce();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_bech32_p2wpkh(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[45] = logic::check_bech32_p2wpkh();
}

// Slots 46-56: tier-2 arithmetic bisect.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_overflowing_add(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[46] = logic::check_arith_overflowing_add();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_overflowing_sub(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[47] = logic::check_arith_overflowing_sub();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_carry_chain_3limb(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[48] = logic::check_arith_carry_chain_3limb();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_widening_mul_pair(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[49] = logic::check_arith_widening_mul_pair();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_mad_lo_u64(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[50] = logic::check_arith_mad_lo_u64();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_mad_hi_u64(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[51] = logic::check_arith_mad_hi_u64();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_mul_wide_u32(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[52] = logic::check_arith_mul_wide_u32();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_mask_blend_true(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[53] = logic::check_arith_mask_blend_true();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_mask_blend_false(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[54] = logic::check_arith_mask_blend_false();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_var_shr_u64(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[55] = logic::check_arith_var_shr_u64();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_var_shl_u64(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[56] = logic::check_arith_var_shl_u64();
}

// Slots 57-58: black_box identity smoking-gun probes.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_blackbox_identity_u64(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[57] = logic::check_arith_blackbox_identity_u64();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_blackbox_identity_u32(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[58] = logic::check_arith_blackbox_identity_u32();
}

// Slot 59: isolated divmod-by-58 — confirms slot 41's crash is
// downstream of the same `mul.hi.u64` codegen bug, not independent.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_base58_div_by_58(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[59] = logic::check_base58_div_by_58();
}

// Slots 60-62: triangulating bisects for the slot 41/43 crash class.
// 60 isolates the bare static-table lookup, 61 isolates iter_mut over
// `&mut [u8; N][..n]`, 62 combines both into the exact shape
// base58_encode_32's final alphabet-encoding loop takes.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_iter_static_table_lookup(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[60] = logic::check_iter_static_table_lookup();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_iter_mut_slice_partial(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[61] = logic::check_iter_mut_slice_partial();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_iter_mut_alphabet_lookup(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[62] = logic::check_iter_mut_alphabet_lookup();
}

// Slot 63: `&[u8]` slice counterpart to slot 60's `&[u8; N]` array
// reference probe. Same shape, one-variable comparison for the
// array-ref-vs-slice discriminator hypothesis.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_iter_static_slice_lookup(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[63] = logic::check_iter_static_slice_lookup();
}

// Slots 64-65: in-suite versions of the cuda-oxide standalone repros
// (divrem_large_const, i128_add_carry_chain). Adding them here means
// every future compiler bump re-validates these specific shapes
// automatically.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_divrem_by_58_pow_5(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[64] = logic::check_arith_divrem_by_58_pow_5();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_i128_chain_add(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[65] = logic::check_arith_i128_chain_add();
}

// Slots 66-68: ports of three cuda-oxide standalone-repro hypotheses
// that aren't covered by any existing in-suite slot.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_base58_limb_divrem(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[66] = logic::check_base58_limb_divrem();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dynamic_index_write(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[67] = logic::check_dynamic_index_write();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_widening_mul_chain_3term(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[68] = logic::check_arith_widening_mul_chain_3term();
}

// Slot 69: base58 Phase A inner-mutate (the only outer-loop phase
// slot 67 doesn't cover and the only phase slot 43's all-zero input
// doesn't exercise — exactly the asymmetry between slot 41/43).
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_base58_inner_mutate_phase(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[69] = logic::check_base58_inner_mutate_phase();
}

// Slots 70-72: curve25519-dalek per-stage bisect for slot 2.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_clamp_integer(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[70] = logic::check_dalek_clamp_integer();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar_round_trip_one(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[71] = logic::check_dalek_scalar_round_trip_one();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_mul_base_scalar_one(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[72] = logic::check_dalek_mul_base_scalar_one();
}

// Slots 73-75: k256 per-stage bisect for slot 4.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_k256_secret_from_bytes_one(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[73] = logic::check_k256_secret_from_bytes_one();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_k256_derive_scalar_one(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[74] = logic::check_k256_derive_scalar_one();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_k256_derive_scalar_two(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[75] = logic::check_k256_derive_scalar_two();
}

// Slots 76-77: unifying-hypothesis probes for the &'static multi-byte
// element bug suspected behind slots 2/4/41/42/71/72/74/75.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_static_u64_array_lookup(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[76] = logic::check_static_u64_array_lookup();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_static_struct_wrapped_u64_lookup(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[77] = logic::check_static_struct_wrapped_u64_lookup();
}

// Slots 78-80: k256 bug-triangulation probes (Bug B in KNOWN_FAILURES).
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_k256_encode_generator(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[78] = logic::check_k256_encode_generator();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_k256_double_generator(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[79] = logic::check_k256_double_generator();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_k256_scalar_one_round_trip(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[80] = logic::check_k256_scalar_one_round_trip();
}

// Slots 81-83: post-v1.46 re-bisect probes.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_arith_u128_imm_shr_52(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[81] = logic::check_arith_u128_imm_shr_52();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_static_depth4_newtype_nesting(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[82] = logic::check_static_depth4_newtype_nesting();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_reverse_range_write(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[83] = logic::check_reverse_range_write();
}

// Slots 84-87: dalek Scalar52 ladder bisect of slot 71's chain
// (using a verbatim port in logic::bisect_scalar52 since dalek's
// backend module is pub(crate)).
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar52_from_bytes(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[84] = logic::check_dalek_scalar52_from_bytes();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar52_montgomery_reduce_r(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[85] = logic::check_dalek_scalar52_montgomery_reduce_r();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar52_mul_internal_then_reduce_one_r(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[86] = logic::check_dalek_scalar52_mul_internal_then_reduce_one_r();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar52_as_bytes_one(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[87] = logic::check_dalek_scalar52_as_bytes_one();
}

// Slots 88-90: Scalar52::sub probes added after slot 86 PASSed
// without the final sub call (a difference from real dalek).
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar52_sub_no_underflow(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[88] = logic::check_dalek_scalar52_sub_no_underflow();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar52_sub_with_underflow(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[89] = logic::check_dalek_scalar52_sub_with_underflow();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar52_montgomery_reduce_with_sub(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[90] = logic::check_dalek_scalar52_montgomery_reduce_with_sub();
}

// Slots 91-93: post-round-2 probes (Index trait dispatch + cross-crate
// minimal probes).
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_index_trait_dispatch(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[91] = logic::check_index_trait_dispatch();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar_one_to_bytes_direct(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[92] = logic::check_dalek_scalar_one_to_bytes_direct();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_k256_affine_generator_encode(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[93] = logic::check_k256_affine_generator_encode();
}

// Slots 94-96: Bug F bisect (subtle::Choice + ConditionallySelectable +
// EncodedPoint construction).
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_subtle_choice_u8_into_bool(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[94] = logic::check_subtle_choice_u8_into_bool();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_subtle_conditional_select_u64(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[95] = logic::check_subtle_conditional_select_u64();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_k256_encoded_point_from_affine_coords(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[96] = logic::check_k256_encoded_point_from_affine_coords();
}

// Slots 97-99: post-round-4 probes (const-idx Index trait + GenericArray).
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_index_trait_const_indices(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[97] = logic::check_index_trait_const_indices();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_generic_array_basic_index(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[98] = logic::check_generic_array_basic_index();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_generic_array_copy_from_slice(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[99] = logic::check_generic_array_copy_from_slice();
}

// Slots 100-102: post-round-5 probes.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_from_affine_coords_replica(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[100] = logic::check_from_affine_coords_replica();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_generic_array_as_slice_last(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[101] = logic::check_generic_array_as_slice_last();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar_round_trip_zero(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[102] = logic::check_dalek_scalar_round_trip_zero();
}

// Slots 103-105: one fresh probe per open bug (Bug-71, Bug-96, Bug-41).
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar_from_bytes_wide_zero(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[103] = logic::check_dalek_scalar_from_bytes_wide_zero();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_field_bytes_into_conversion(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[104] = logic::check_field_bytes_into_conversion();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_base58_min_nonzero(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[105] = logic::check_base58_min_nonzero();
}

// Slots 106-107: breakthrough probes.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_named_field_struct_return(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[106] = logic::check_named_field_struct_return();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_base58_handrolled_no_seq(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[107] = logic::check_base58_handrolled_no_seq();
}

// Slots 108-109: post-round-7 probes.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_slice_reverse_partial(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[108] = logic::check_slice_reverse_partial();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar_eq_zero(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[109] = logic::check_dalek_scalar_eq_zero();
}

// Slots 110-112: post-round-8 probes.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_generic_array_copy_from_ga_source(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[110] = logic::check_generic_array_copy_from_ga_source();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_zero_eq_zero(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[111] = logic::check_dalek_zero_eq_zero();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_from_canonical_zero(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[112] = logic::check_dalek_from_canonical_zero();
}

// Slots 113-115: post-round-9 zero-input verbatim-port ladder.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar52_from_bytes_zero(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[113] = logic::check_dalek_scalar52_from_bytes_zero();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar52_mul_internal_zero(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[114] = logic::check_dalek_scalar52_mul_internal_zero();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar52_montgomery_reduce_zero(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[115] = logic::check_dalek_scalar52_montgomery_reduce_zero();
}

// Slots 116-117: post-round-10 probes.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_scalar52_as_bytes_zero(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[116] = logic::check_dalek_scalar52_as_bytes_zero();
}
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_dalek_reduce_pipeline_zero(results_ptr: *mut u32) {
    let results = unsafe { core::slice::from_raw_parts_mut(results_ptr, logic::SELF_TEST_NUM_CHECKS) };
    results[117] = logic::check_dalek_reduce_pipeline_zero();
}
