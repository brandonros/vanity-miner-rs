//! bitcoin checks; one result per entry of `CHECKS`, in order.

#![no_std]
#![feature(abi_ptx)]

// Links the device runtime.
use kernel_common as _;
use logic::self_test::bitcoin::{CHECKS, run};

#[unsafe(no_mangle)]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "ptx-kernel" fn kernel_self_test_bitcoin(results: *mut u32) {
    run(unsafe { core::slice::from_raw_parts_mut(results, CHECKS.len()) });
}
