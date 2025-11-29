/// Macro to handle the common pattern when a vanity match is found in a kernel.
///
/// Handles:
/// 1. Reconstructing found_matches slice from raw pointer
/// 2. Atomic check for first-find (only first match copies data)
/// 3. Copying result fields to device buffers
/// 4. Incrementing match counter
///
/// Copy syntax:
/// - `src => ptr, len;`                      - full slice copy
/// - `partial: src, dyn_len => ptr, max;`    - partial slice copy
/// - `scalar: value => ptr;`                 - scalar assignment (single element)
#[macro_export]
macro_rules! handle_match {
    (
        thread_idx: $thread_idx:expr,
        found_matches_ptr: $found_matches_ptr:expr,
        copies: [ $( $copy:tt )* ],
        found_thread_idx_ptr: $found_thread_idx_ptr:expr $(,)?
    ) => {{
        let found_matches_slice = unsafe { core::slice::from_raw_parts_mut($found_matches_ptr, 1) };
        let found_matches = &mut found_matches_slice[0];

        // If first find, copy results to device buffers
        if unsafe { $crate::atomic::atomic_add_u32(found_matches, 0) } == 0 {
            handle_match!(@copies $($copy)*);

            let found_thread_idx_slice = unsafe { core::slice::from_raw_parts_mut($found_thread_idx_ptr, 1) };
            found_thread_idx_slice[0] = $thread_idx as u32;
        }

        // Increment number of found matches
        unsafe { $crate::atomic::atomic_add_u32(found_matches, 1) };
    }};

    // Full slice copy: src => ptr, len;
    (@copies $src:expr => $dst_ptr:expr, $len:expr ; $($rest:tt)*) => {
        let dst = unsafe { core::slice::from_raw_parts_mut($dst_ptr, $len) };
        dst.copy_from_slice(&$src);
        handle_match!(@copies $($rest)*);
    };

    // Partial slice copy: partial: src, dyn_len => ptr, max_len;
    (@copies partial: $src:expr, $dyn_len:expr => $dst_ptr:expr, $max_len:expr ; $($rest:tt)*) => {
        let dst = unsafe { core::slice::from_raw_parts_mut($dst_ptr, $max_len) };
        dst[..$dyn_len].copy_from_slice(&$src[..$dyn_len]);
        handle_match!(@copies $($rest)*);
    };

    // Scalar assignment: scalar: value => ptr;
    (@copies scalar: $value:expr => $dst_ptr:expr ; $($rest:tt)*) => {
        let dst = unsafe { core::slice::from_raw_parts_mut($dst_ptr, 1) };
        dst[0] = $value;
        handle_match!(@copies $($rest)*);
    };

    // Base case - no more copies
    (@copies) => {};
}
