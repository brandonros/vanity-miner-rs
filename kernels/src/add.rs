use crate::utilities;

#[unsafe(no_mangle)]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_add(
    input_a_ptr: *const f32,
    input_b_ptr: *const f32, 
    output_ptr: *mut f32,
) {
    let thread_idx = utilities::get_thread_idx();
    
    // Perform the addition
    let a = unsafe { *input_a_ptr.add(thread_idx) };
    let b = unsafe { *input_b_ptr.add(thread_idx) };
    let result = a + b;
    
    unsafe {
        *output_ptr.add(thread_idx) = result;
    }
}
