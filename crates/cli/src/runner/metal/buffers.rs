use llvm_metal_runtime::Buffer;
use logic::search::device_record::DeviceRecord;
pub(super) fn bytes<T: DeviceRecord>(value: &T) -> &[u8] {
    // SAFETY: DeviceRecord guarantees initialized padding-free storage.
    unsafe {
        std::slice::from_raw_parts(std::ptr::from_ref(value).cast(), std::mem::size_of::<T>())
    }
}
pub(super) fn record<T: DeviceRecord>(bytes: &[u8]) -> T {
    assert_eq!(bytes.len(), std::mem::size_of::<T>());
    // SAFETY: exact-size bytes, any bit pattern valid; alignment is not assumed.
    unsafe { bytes.as_ptr().cast::<T>().read_unaligned() }
}
pub(super) fn guarded(size: usize) -> Buffer {
    Buffer {
        bytes: vec![0xa5; size + 512],
        offset: 256,
    }
}
