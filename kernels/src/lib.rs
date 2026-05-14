#[macro_use]
mod match_handler;

use cuda_device::{cuda_module, device, kernel};

#[cuda_module]
pub mod kernels {
    use super::*;
    use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
    use cuda_device::thread;

    /// 1-D global thread index. `#[device]` (not `#[inline]`) because this
    /// helper *calls* a cuda-oxide intrinsic (`thread::index_1d`). The
    /// `#[cuda_module]` macro rewrites intrinsic call sites only inside
    /// `#[kernel]` and `#[device]` items; under `#[inline]` rustc folds the
    /// intrinsic's `unreachable!()` stub body into every caller before the
    /// rewrite pass runs, and the optimizer then DCEs the kernel down to a
    /// single `panic_fmt` — emitting PTX whose only operation is `exit;`.
    #[device]
    pub fn get_thread_idx() -> usize {
        thread::index_1d().get()
    }

    /// Device-scope relaxed atomic add over a u32 location borrowed from a
    /// slice element. `#[device]` for the same reason as `get_thread_idx`:
    /// the body calls intrinsics (`DeviceAtomicU32::from_ptr`, `.fetch_add`)
    /// that the macro must rewrite at *this* call site, not at the kernel.
    ///
    /// Not marked `unsafe fn` even though the body does an unsafe op:
    /// cuda-oxide's `#[device]` macro generates a safe wrapper that doesn't
    /// propagate the `unsafe` modifier, so an `unsafe fn` declaration here
    /// would produce a wrapper that fails to compile (E0133). Given the
    /// `&mut u32` argument type, the API is in fact callable safely —
    /// exclusive aliasing is guaranteed by the borrow.
    #[device]
    pub fn atomic_add_u32(address: &mut u32, val: u32) -> u32 {
        unsafe {
            DeviceAtomicU32::from_ptr(address as *mut u32)
                .fetch_add(val, AtomicOrdering::Relaxed)
        }
    }

    /// Bitcoin vanity search kernel. Output slice lengths:
    /// matches:1 priv:32 pub:33 hash:20 enc:64 enc_len:1 thread_idx:1
    #[cfg(feature = "kernel_bitcoin")]
    #[kernel]
    #[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
    pub unsafe fn kernel_find_bitcoin_vanity_private_key(
        vanity_prefix: &[u8],
        vanity_suffix: &[u8],
        rng_seed: u64,
        found_matches: &mut [u32],
        found_private_key: &mut [u8],
        found_public_key: &mut [u8],
        found_public_key_hash: &mut [u8],
        found_encoded_public_key: &mut [u8],
        found_encoded_len: &mut [u32],
        found_thread_idx: &mut [u32],
    ) {
        let thread_idx = get_thread_idx();
        let request = logic::BitcoinVanityKeyRequest {
            prefix: vanity_prefix,
            suffix: vanity_suffix,
            thread_idx,
            rng_seed,
        };

        let result = logic::generate_and_check_bitcoin_vanity_key(&request);

        if result.matches {
            handle_match! {
                thread_idx: thread_idx,
                found_matches: found_matches,
                copies: [
                    result.private_key => found_private_key;
                    result.public_key => found_public_key;
                    result.public_key_hash => found_public_key_hash;
                    partial: result.encoded_public_key, result.encoded_len => found_encoded_public_key;
                    scalar: result.encoded_len as u32 => found_encoded_len;
                ],
                found_thread_idx: found_thread_idx,
            }
        }
    }

    /// Ethereum vanity search kernel. Output slice lengths:
    /// matches:1 priv:32 pub:64 address:20 thread_idx:1
    #[cfg(feature = "kernel_ethereum")]
    #[kernel]
    #[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
    pub unsafe fn kernel_find_ethereum_vanity_private_key(
        vanity_prefix: &[u8],
        vanity_suffix: &[u8],
        rng_seed: u64,
        found_matches: &mut [u32],
        found_private_key: &mut [u8],
        found_public_key: &mut [u8],
        found_address: &mut [u8],
        found_thread_idx: &mut [u32],
    ) {
        let thread_idx = get_thread_idx();
        let request = logic::EthereumVanityKeyRequest {
            prefix: vanity_prefix,
            suffix: vanity_suffix,
            thread_idx,
            rng_seed,
        };

        let result = logic::generate_and_check_ethereum_vanity_key(&request);

        if result.matches {
            handle_match! {
                thread_idx: thread_idx,
                found_matches: found_matches,
                copies: [
                    result.private_key => found_private_key;
                    result.public_key => found_public_key;
                    result.address => found_address;
                ],
                found_thread_idx: found_thread_idx,
            }
        }
    }

    /// Solana vanity search kernel. Output slice lengths:
    /// matches:1 priv:32 pub:32 bs58:64 thread_idx:1
    #[cfg(feature = "kernel_solana")]
    #[kernel]
    #[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
    pub unsafe fn kernel_find_solana_vanity_private_key(
        vanity_prefix: &[u8],
        vanity_suffix: &[u8],
        rng_seed: u64,
        found_matches: &mut [u32],
        found_private_key: &mut [u8],
        found_public_key: &mut [u8],
        found_bs58_encoded_public_key: &mut [u8],
        found_thread_idx: &mut [u32],
    ) {
        let thread_idx = get_thread_idx();
        let request = logic::SolanaVanityKeyRequest {
            prefix: vanity_prefix,
            suffix: vanity_suffix,
            thread_idx,
            rng_seed,
        };

        let result = logic::generate_and_check_solana_vanity_key(&request);

        if result.matches {
            handle_match! {
                thread_idx: thread_idx,
                found_matches: found_matches,
                copies: [
                    result.private_key => found_private_key;
                    result.public_key => found_public_key;
                    result.encoded_public_key => found_bs58_encoded_public_key;
                ],
                found_thread_idx: found_thread_idx,
            }
        }
    }

    /// Race-tolerant write of a better hash. Multiple threads may overwrite
    /// each other within a single launch; the host keeps the global best
    /// across launches. `#[device]` so the call to `atomic_add_u32` (itself
    /// a `#[device]` wrapper around a cuda-oxide atomic intrinsic) is
    /// preserved instead of being folded into the kernel before the macro's
    /// rewrite pass runs.
    #[cfg(feature = "kernel_shallenge")]
    #[device]
    fn handle_shallenge_match_found(
        result: logic::ShallengeResult,
        thread_idx: usize,
        found_matches: &mut [u32],
        found_hash: &mut [u8],
        found_nonce: &mut [u8],
        found_nonce_len: &mut [usize],
        found_thread_idx: &mut [u32],
    ) {
        found_hash.copy_from_slice(&result.hash);
        found_nonce.copy_from_slice(&result.nonce);
        found_nonce_len[0] = result.nonce_len;
        found_thread_idx[0] = thread_idx as u32;
        atomic_add_u32(&mut found_matches[0], 1);
    }

    /// Shallenge search kernel. Slice lengths:
    /// target_hash:32 matches:1 hash:32 nonce:64 nonce_len:1 thread_idx:1
    #[cfg(feature = "kernel_shallenge")]
    #[kernel]
    #[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
    pub unsafe fn kernel_find_better_shallenge_nonce(
        username: &[u8],
        target_hash: &[u8],
        rng_seed: u64,
        found_matches: &mut [u32],
        found_hash: &mut [u8],
        found_nonce: &mut [u8],
        found_nonce_len: &mut [usize],
        found_thread_idx: &mut [u32],
    ) {
        let thread_idx = get_thread_idx();
        let username_len = username.len();
        let target_hash_array: &[u8; 32] =
            unsafe { &*(target_hash.as_ptr() as *const [u8; 32]) };

        let request = logic::ShallengeRequest {
            username,
            username_len,
            target_hash: target_hash_array,
            thread_idx,
            rng_seed,
        };

        let result = logic::generate_and_check_shallenge(&request);

        if result.is_better {
            handle_shallenge_match_found(
                result,
                thread_idx,
                found_matches,
                found_hash,
                found_nonce,
                found_nonce_len,
                found_thread_idx,
            );
        }
    }

    /// Self-test plumbing probe: writes `results[0] = 1` and nothing else.
    /// Launched before the per-slot kernels to isolate "PTX load + launch +
    /// DtoH copy" health from any logic body. If this fails, the bug isn't
    /// in `logic::*` — it's in the kernel-load / launch path.
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_stub(results: &mut [u32]) {
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

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_primitive_xoroshiro(results: &mut [u32]) {
        results[0] = logic::check_primitive_xoroshiro();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_primitive_sha512(results: &mut [u32]) {
        results[1] = logic::check_primitive_sha512();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_primitive_ed25519(results: &mut [u32]) {
        results[2] = logic::check_primitive_ed25519();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_primitive_base58(results: &mut [u32]) {
        results[3] = logic::check_primitive_base58();
    }

    // Slots 4-9: non-solana primitive bisect — secp256k1 (compressed +
    // uncompressed), keccak256, ripemd160, sha256 (fixed-32 + variable).
    // These are the primitives the bitcoin / ethereum / shallenge / WIF
    // pipelines compose; isolating each one means a fault here pinpoints
    // the broken primitive before the composed kernels (slots 10+) would
    // have inlined it.

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_primitive_secp256k1_compressed(results: &mut [u32]) {
        results[4] = logic::check_primitive_secp256k1_compressed();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_primitive_secp256k1_uncompressed(results: &mut [u32]) {
        results[5] = logic::check_primitive_secp256k1_uncompressed();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_primitive_keccak256(results: &mut [u32]) {
        results[6] = logic::check_primitive_keccak256();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_primitive_ripemd160(results: &mut [u32]) {
        results[7] = logic::check_primitive_ripemd160();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_primitive_sha256_32(results: &mut [u32]) {
        results[8] = logic::check_primitive_sha256_32();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_primitive_sha256_variable(results: &mut [u32]) {
        results[9] = logic::check_primitive_sha256_variable();
    }

    // Slots 10-30: composed-subsystem KAT checks.

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_solana_priv(results: &mut [u32]) {
        results[10] = logic::check_solana_priv();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_solana_pub(results: &mut [u32]) {
        results[11] = logic::check_solana_pub();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_solana_encoded(results: &mut [u32]) {
        results[12] = logic::check_solana_encoded();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_ethereum_priv(results: &mut [u32]) {
        results[13] = logic::check_ethereum_priv();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_ethereum_pub(results: &mut [u32]) {
        results[14] = logic::check_ethereum_pub();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_ethereum_address(results: &mut [u32]) {
        results[15] = logic::check_ethereum_address();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_bitcoin_priv(results: &mut [u32]) {
        results[16] = logic::check_bitcoin_priv();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_bitcoin_pub(results: &mut [u32]) {
        results[17] = logic::check_bitcoin_pub();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_bitcoin_pkh(results: &mut [u32]) {
        results[18] = logic::check_bitcoin_pkh();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_bitcoin_encoded(results: &mut [u32]) {
        results[19] = logic::check_bitcoin_encoded();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_bitcoin_matches(results: &mut [u32]) {
        results[20] = logic::check_bitcoin_matches();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_wif_compressed_mainnet(results: &mut [u32]) {
        results[21] = logic::check_wif_compressed_mainnet();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_wif_uncompressed_mainnet(results: &mut [u32]) {
        results[22] = logic::check_wif_uncompressed_mainnet();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_wif_compressed_testnet(results: &mut [u32]) {
        results[23] = logic::check_wif_compressed_testnet();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_wif_uncompressed_testnet(results: &mut [u32]) {
        results[24] = logic::check_wif_uncompressed_testnet();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_shallenge_hash(results: &mut [u32]) {
        results[25] = logic::check_shallenge_hash();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_shallenge_nonce_len(results: &mut [u32]) {
        results[26] = logic::check_shallenge_nonce_len();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_shallenge_is_better(results: &mut [u32]) {
        results[27] = logic::check_shallenge_is_better();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_compare_hashes_lt(results: &mut [u32]) {
        results[28] = logic::check_compare_hashes_lt();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_compare_hashes_gt(results: &mut [u32]) {
        results[29] = logic::check_compare_hashes_gt();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_compare_hashes_eq(results: &mut [u32]) {
        results[30] = logic::check_compare_hashes_eq();
    }

    // Slots 31-40: raw-arithmetic bisect — one PTX op per kernel. Each
    // baselines against a host-rustc `const`-evaluated expected value, so a
    // PASS on CPU + FAIL on GPU isolates the codegen bug to that exact op.

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_u32_div_var(results: &mut [u32]) {
        results[31] = logic::check_arith_u32_div_var();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_u32_div_const(results: &mut [u32]) {
        results[32] = logic::check_arith_u32_div_const();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_u64_div_var(results: &mut [u32]) {
        results[33] = logic::check_arith_u64_div_var();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_u64_div_const(results: &mut [u32]) {
        results[34] = logic::check_arith_u64_div_const();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_u32_rem_var(results: &mut [u32]) {
        results[35] = logic::check_arith_u32_rem_var();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_u64_rem_var(results: &mut [u32]) {
        results[36] = logic::check_arith_u64_rem_var();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_u32_mul_lo(results: &mut [u32]) {
        results[37] = logic::check_arith_u32_mul_lo();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_u64_mul_lo(results: &mut [u32]) {
        results[38] = logic::check_arith_u64_mul_lo();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_u64_mul_hi(results: &mut [u32]) {
        results[39] = logic::check_arith_u64_mul_hi();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_u128_mul(results: &mut [u32]) {
        results[40] = logic::check_arith_u128_mul();
    }

    // Slots 41-45: composed-primitive sub-bisects.

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_base58_var_len(results: &mut [u32]) {
        results[41] = logic::check_base58_var_len();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_base58_var_len_leading_zero(results: &mut [u32]) {
        results[42] = logic::check_base58_var_len_leading_zero();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_base58_all_zeros(results: &mut [u32]) {
        results[43] = logic::check_base58_all_zeros();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_xoroshiro_base64_nonce(results: &mut [u32]) {
        results[44] = logic::check_xoroshiro_base64_nonce();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_bech32_p2wpkh(results: &mut [u32]) {
        results[45] = logic::check_bech32_p2wpkh();
    }

    // Slots 46-56: tier-2 arithmetic bisect.

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_overflowing_add(results: &mut [u32]) {
        results[46] = logic::check_arith_overflowing_add();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_overflowing_sub(results: &mut [u32]) {
        results[47] = logic::check_arith_overflowing_sub();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_carry_chain_3limb(results: &mut [u32]) {
        results[48] = logic::check_arith_carry_chain_3limb();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_widening_mul_pair(results: &mut [u32]) {
        results[49] = logic::check_arith_widening_mul_pair();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_mad_lo_u64(results: &mut [u32]) {
        results[50] = logic::check_arith_mad_lo_u64();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_mad_hi_u64(results: &mut [u32]) {
        results[51] = logic::check_arith_mad_hi_u64();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_mul_wide_u32(results: &mut [u32]) {
        results[52] = logic::check_arith_mul_wide_u32();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_mask_blend_true(results: &mut [u32]) {
        results[53] = logic::check_arith_mask_blend_true();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_mask_blend_false(results: &mut [u32]) {
        results[54] = logic::check_arith_mask_blend_false();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_var_shr_u64(results: &mut [u32]) {
        results[55] = logic::check_arith_var_shr_u64();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_var_shl_u64(results: &mut [u32]) {
        results[56] = logic::check_arith_var_shl_u64();
    }

    // Slots 57-58: black_box identity smoking-gun probes.

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_blackbox_identity_u64(results: &mut [u32]) {
        results[57] = logic::check_arith_blackbox_identity_u64();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_blackbox_identity_u32(results: &mut [u32]) {
        results[58] = logic::check_arith_blackbox_identity_u32();
    }

    // Slot 59: isolated divmod-by-58 — confirms slot 41's crash is
    // downstream of the same `mul.hi.u64` codegen bug, not independent.

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_base58_div_by_58(results: &mut [u32]) {
        results[59] = logic::check_base58_div_by_58();
    }

    // Slots 60-62: triangulating bisects for the slot 41/43 crash class.
    // 60 isolates the bare static-table lookup, 61 isolates iter_mut over
    // `&mut [u8; N][..n]`, 62 combines both into the exact shape
    // base58_encode_32's final alphabet-encoding loop takes.

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_iter_static_table_lookup(results: &mut [u32]) {
        results[60] = logic::check_iter_static_table_lookup();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_iter_mut_slice_partial(results: &mut [u32]) {
        results[61] = logic::check_iter_mut_slice_partial();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_iter_mut_alphabet_lookup(results: &mut [u32]) {
        results[62] = logic::check_iter_mut_alphabet_lookup();
    }

    // Slot 63: `&[u8]` slice counterpart to slot 60's `&[u8; N]` array
    // reference probe. Same shape, one-variable comparison for the
    // array-ref-vs-slice discriminator hypothesis.

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_iter_static_slice_lookup(results: &mut [u32]) {
        results[63] = logic::check_iter_static_slice_lookup();
    }

    // Slots 64-65: in-suite versions of the cuda-oxide standalone repros
    // (divrem_large_const, i128_add_carry_chain). Adding them here means
    // every future compiler bump re-validates these specific shapes
    // automatically.

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_divrem_by_58_pow_5(results: &mut [u32]) {
        results[64] = logic::check_arith_divrem_by_58_pow_5();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_i128_chain_add(results: &mut [u32]) {
        results[65] = logic::check_arith_i128_chain_add();
    }

    // Slots 66-68: ports of three cuda-oxide standalone-repro hypotheses
    // that aren't covered by any existing in-suite slot.

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_base58_limb_divrem(results: &mut [u32]) {
        results[66] = logic::check_base58_limb_divrem();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dynamic_index_write(results: &mut [u32]) {
        results[67] = logic::check_dynamic_index_write();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_widening_mul_chain_3term(results: &mut [u32]) {
        results[68] = logic::check_arith_widening_mul_chain_3term();
    }

    // Slot 69: base58 Phase A inner-mutate (the only outer-loop phase
    // slot 67 doesn't cover and the only phase slot 43's all-zero input
    // doesn't exercise — exactly the asymmetry between slot 41/43).
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_base58_inner_mutate_phase(results: &mut [u32]) {
        results[69] = logic::check_base58_inner_mutate_phase();
    }

    // Slots 70-72: curve25519-dalek per-stage bisect for slot 2.
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dalek_clamp_integer(results: &mut [u32]) {
        results[70] = logic::check_dalek_clamp_integer();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dalek_scalar_round_trip_one(results: &mut [u32]) {
        results[71] = logic::check_dalek_scalar_round_trip_one();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dalek_mul_base_scalar_one(results: &mut [u32]) {
        results[72] = logic::check_dalek_mul_base_scalar_one();
    }

    // Slots 73-75: k256 per-stage bisect for slot 4.
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_k256_secret_from_bytes_one(results: &mut [u32]) {
        results[73] = logic::check_k256_secret_from_bytes_one();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_k256_derive_scalar_one(results: &mut [u32]) {
        results[74] = logic::check_k256_derive_scalar_one();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_k256_derive_scalar_two(results: &mut [u32]) {
        results[75] = logic::check_k256_derive_scalar_two();
    }

    // Slots 76-77: unifying-hypothesis probes for the &'static multi-byte
    // element bug suspected behind slots 2/4/41/42/71/72/74/75.
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_static_u64_array_lookup(results: &mut [u32]) {
        results[76] = logic::check_static_u64_array_lookup();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_static_struct_wrapped_u64_lookup(results: &mut [u32]) {
        results[77] = logic::check_static_struct_wrapped_u64_lookup();
    }

    // Slots 78-80: k256 bug-triangulation probes (Bug B in KNOWN_FAILURES).
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_k256_encode_generator(results: &mut [u32]) {
        results[78] = logic::check_k256_encode_generator();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_k256_double_generator(results: &mut [u32]) {
        results[79] = logic::check_k256_double_generator();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_k256_scalar_one_round_trip(results: &mut [u32]) {
        results[80] = logic::check_k256_scalar_one_round_trip();
    }

    // Slots 81-83: post-v1.46 re-bisect probes.
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_arith_u128_imm_shr_52(results: &mut [u32]) {
        results[81] = logic::check_arith_u128_imm_shr_52();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_static_depth4_newtype_nesting(results: &mut [u32]) {
        results[82] = logic::check_static_depth4_newtype_nesting();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_reverse_range_write(results: &mut [u32]) {
        results[83] = logic::check_reverse_range_write();
    }

    // Slots 84-87: dalek Scalar52 ladder bisect of slot 71's chain
    // (using a verbatim port in logic::bisect_scalar52 since dalek's
    // backend module is pub(crate)).
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dalek_scalar52_from_bytes(results: &mut [u32]) {
        results[84] = logic::check_dalek_scalar52_from_bytes();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dalek_scalar52_montgomery_reduce_r(results: &mut [u32]) {
        results[85] = logic::check_dalek_scalar52_montgomery_reduce_r();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dalek_scalar52_mul_internal_then_reduce_one_r(results: &mut [u32]) {
        results[86] = logic::check_dalek_scalar52_mul_internal_then_reduce_one_r();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dalek_scalar52_as_bytes_one(results: &mut [u32]) {
        results[87] = logic::check_dalek_scalar52_as_bytes_one();
    }

    // Slots 88-90: Scalar52::sub probes added after slot 86 PASSed
    // without the final sub call (a difference from real dalek).
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dalek_scalar52_sub_no_underflow(results: &mut [u32]) {
        results[88] = logic::check_dalek_scalar52_sub_no_underflow();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dalek_scalar52_sub_with_underflow(results: &mut [u32]) {
        results[89] = logic::check_dalek_scalar52_sub_with_underflow();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dalek_scalar52_montgomery_reduce_with_sub(results: &mut [u32]) {
        results[90] = logic::check_dalek_scalar52_montgomery_reduce_with_sub();
    }

    // Slots 91-93: post-round-2 probes (Index trait dispatch + cross-crate
    // minimal probes).
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_index_trait_dispatch(results: &mut [u32]) {
        results[91] = logic::check_index_trait_dispatch();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dalek_scalar_one_to_bytes_direct(results: &mut [u32]) {
        results[92] = logic::check_dalek_scalar_one_to_bytes_direct();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_k256_affine_generator_encode(results: &mut [u32]) {
        results[93] = logic::check_k256_affine_generator_encode();
    }

    // Slots 94-96: Bug F bisect (subtle::Choice + ConditionallySelectable +
    // EncodedPoint construction).
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_subtle_choice_u8_into_bool(results: &mut [u32]) {
        results[94] = logic::check_subtle_choice_u8_into_bool();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_subtle_conditional_select_u64(results: &mut [u32]) {
        results[95] = logic::check_subtle_conditional_select_u64();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_k256_encoded_point_from_affine_coords(results: &mut [u32]) {
        results[96] = logic::check_k256_encoded_point_from_affine_coords();
    }

    // Slots 97-99: post-round-4 probes (const-idx Index trait + GenericArray).
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_index_trait_const_indices(results: &mut [u32]) {
        results[97] = logic::check_index_trait_const_indices();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_generic_array_basic_index(results: &mut [u32]) {
        results[98] = logic::check_generic_array_basic_index();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_generic_array_copy_from_slice(results: &mut [u32]) {
        results[99] = logic::check_generic_array_copy_from_slice();
    }

    // Slots 100-102: post-round-5 probes.
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_from_affine_coords_replica(results: &mut [u32]) {
        results[100] = logic::check_from_affine_coords_replica();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_generic_array_as_slice_last(results: &mut [u32]) {
        results[101] = logic::check_generic_array_as_slice_last();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dalek_scalar_round_trip_zero(results: &mut [u32]) {
        results[102] = logic::check_dalek_scalar_round_trip_zero();
    }

    // Slots 103-105: one fresh probe per open bug (Bug-71, Bug-96, Bug-41).
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_dalek_scalar_from_bytes_wide_zero(results: &mut [u32]) {
        results[103] = logic::check_dalek_scalar_from_bytes_wide_zero();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_field_bytes_into_conversion(results: &mut [u32]) {
        results[104] = logic::check_field_bytes_into_conversion();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_base58_min_nonzero(results: &mut [u32]) {
        results[105] = logic::check_base58_min_nonzero();
    }

    // Slots 106-107: breakthrough probes.
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_named_field_struct_return(results: &mut [u32]) {
        results[106] = logic::check_named_field_struct_return();
    }

    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_base58_handrolled_no_seq(results: &mut [u32]) {
        results[107] = logic::check_base58_handrolled_no_seq();
    }
}
