use alloc::alloc::{GlobalAlloc, Layout};
use core::ffi::c_void;

unsafe extern "C" {
    // implicitly defined by cuda.
    pub fn malloc(size: usize) -> *mut c_void;

    pub fn free(ptr: *mut c_void);
}

pub struct CUDAAllocator;

unsafe impl GlobalAlloc for CUDAAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { malloc(layout.size()) as *mut u8 }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        unsafe { free(ptr as *mut _); }
    }
}

#[global_allocator]
pub static GLOBAL_ALLOCATOR: CUDAAllocator = CUDAAllocator;
