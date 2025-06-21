unsafe extern "C" {
    fn __nvvm_atomic_add_global_i32(address: *mut u32, val: u32) -> u32;
}

pub unsafe fn atomic_add_u32(address: *mut u32, val: u32) -> u32 {
    unsafe { __nvvm_atomic_add_global_i32(address, val) }
}
