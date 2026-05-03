use core::sync::atomic::Ordering;

pub unsafe fn atomic_add_u32(address: *mut u32, val: u32) -> u32 {
    unsafe {
        cuda_std::atomic::mid::atomic_fetch_add_u32_device(address, Ordering::Relaxed, val)
    }
}
