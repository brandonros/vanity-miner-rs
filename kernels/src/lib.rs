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
    /// Launched before `kernel_self_test` to isolate "PTX load + launch +
    /// DtoH copy" health from the much larger logic body. If this fails but
    /// the regular self-test would have failed too, the bug is in the
    /// plumbing, not in `logic::run_self_test`.
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test_stub(results: &mut [u32]) {
        results[0] = 1;
    }

    /// Self-test kernel: runs known-answer tests for every logic primitive
    /// against externally-validated expected values, writing pass(1)/fail(0)
    /// per check into the results buffer. Launch as 1×1 grid×block. If any
    /// slot reads 0 on the host, PTX codegen has diverged from CPU behavior.
    /// The body lives in `logic::run_self_test` so CPU and GPU run the
    /// same code; see `logic::SELF_TEST_LABELS` for slot semantics.
    #[cfg(feature = "kernel_self_test")]
    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kernel_self_test(results: &mut [u32]) {
        logic::run_self_test(results);
    }
}
