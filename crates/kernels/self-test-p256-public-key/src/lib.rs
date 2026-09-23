//! p256 public key checks; one result per entry of `CHECKS`, in order.

#![no_std]
#![feature(abi_ptx)]

// Links the device runtime.
use kernel_common as _;
use logic::self_test::p256_public_key::{CHECKS, run};

#[unsafe(no_mangle)]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "ptx-kernel" fn kernel_self_test_p256_public_key(results: *mut u32) {
    run(unsafe { core::slice::from_raw_parts_mut(results, CHECKS.len()) });
}
