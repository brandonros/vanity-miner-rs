//! ethereum checks; result indices come from the shared registry.

#![no_std]
#![feature(abi_ptx)]

// Links the device runtime.
use kernel_common as _;

#[unsafe(no_mangle)]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "ptx-kernel" fn kernel_self_test_ethereum(results_ptr: *mut u32) {
    let results = unsafe {
        core::slice::from_raw_parts_mut(results_ptr, logic::self_test::SELF_TEST_NUM_CHECKS)
    };
    logic::self_test::runners::ethereum::run(results);
}
