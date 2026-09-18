//! Shared launch header for persistent per-lane RSA mining tasks.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Launch {
    pub start: u64,
    pub capacity: u32,
    pub steps: u32,
}
// SAFETY: repr(C), initialized integers only, no padding (checked in tests).
unsafe impl logic::search::device_record::DeviceRecord for Launch {}
pub const INTERFACE: &str = include_str!("../kernel.interface.json");
#[cfg(test)]
mod tests {
    #[test]
    fn layout() {
        assert_eq!(core::mem::size_of::<super::Launch>(), 16);
        assert_eq!(core::mem::align_of::<super::Launch>(), 8);
    }
}
