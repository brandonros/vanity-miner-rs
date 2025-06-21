use core::panic::PanicInfo;

unsafe extern "C" {
    fn __nvvm_trap() -> !;
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    unsafe { __nvvm_trap() };
}
