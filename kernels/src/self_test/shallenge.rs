//! shallenge checks; result indices come from the shared registry.
use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_shallenge(results_ptr: *mut u32) {
    let results = unsafe {
        core::slice::from_raw_parts_mut(results_ptr, logic::self_test::SELF_TEST_NUM_CHECKS)
    };
    logic::self_test::runners::shallenge::run_device(results);
}
