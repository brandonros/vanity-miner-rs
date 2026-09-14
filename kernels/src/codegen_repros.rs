//! Small runtime-input extractions from observed miner failures. One thread per launch.
//! Kept separate from the stable 118 known-answer slots: these return raw outputs.
use cuda_std::prelude::*;

/// Isolate the real Shallenge nonce generator from hashing and match handling.
/// input: [thread index, seed, length <= 64]; output: 64 writable bytes.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_repro_nonce_sequence(input: *const u64, output: *mut u64) {
    let index = unsafe { *input } as usize;
    let seed = unsafe { *input.add(1) };
    let length = (unsafe { *input.add(2) }).min(64) as usize;
    let bytes = unsafe { core::slice::from_raw_parts_mut(output.cast::<u8>(), length) };
    logic::search::xoroshiro::generate_base64_nonce(index, seed, bytes);
}
