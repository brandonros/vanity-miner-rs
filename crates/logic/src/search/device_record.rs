//! Fixed-layout records copied between host and device.

/// A record safe to copy verbatim across the device ABI.
///
/// # Safety
/// Implementors must have a stable layout, no padding or pointers, and accept
/// every bit pattern. All fields must be initialized before transfer.
pub unsafe trait DeviceRecord: Copy + llvm_metal_kernel::DeviceLayout {}

// SAFETY: byte arrays have no padding and accept every bit pattern.
unsafe impl<const N: usize> DeviceRecord for [u8; N] {}
