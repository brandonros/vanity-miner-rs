/// Common copy pattern when a vanity match is found in a kernel.
///
/// Output buffers arrive as `&mut [T]` slices (the `#[cuda_module]` macro
/// unpacks the (ptr, len) pair on the host side). Single-element writes,
/// range indexing, and `copy_from_slice` all compile through cuda-oxide.
///
/// Copy syntax:
/// - `src => dst;`                          - full slice copy
/// - `partial: src, dyn_len => dst;`        - copy first `dyn_len` bytes
/// - `scalar: value => dst;`                - assign to dst[0]
#[macro_export]
macro_rules! handle_match {
    (
        thread_idx: $thread_idx:expr,
        found_matches: $found_matches_slice:expr,
        copies: [ $( $copy:tt )* ],
        found_thread_idx: $found_thread_idx_slice:expr $(,)?
    ) => {{
        let found_matches_slice: &mut [u32] = $found_matches_slice;
        let found_thread_idx_slice: &mut [u32] = $found_thread_idx_slice;

        // First-write-wins: only the first thread to observe a zero counter
        // commits the result fields. Other matches still increment the
        // counter so the host sees a non-zero value. `atomic_add_u32` is
        // unqualified so it resolves at the call site — every invocation
        // happens inside the cuda_module mod where it is defined. It is a
        // safe `pub fn` because cuda-oxide's `#[device]` macro doesn't
        // propagate `unsafe fn` to the generated wrapper.
        if atomic_add_u32(&mut found_matches_slice[0], 0) == 0 {
            handle_match!(@copies $($copy)*);
            found_thread_idx_slice[0] = $thread_idx as u32;
        }

        atomic_add_u32(&mut found_matches_slice[0], 1);
    }};

    // Full slice copy: src => dst;
    (@copies $src:expr => $dst:expr ; $($rest:tt)*) => {
        ($dst).copy_from_slice(&$src);
        handle_match!(@copies $($rest)*);
    };

    // Partial slice copy: partial: src, dyn_len => dst;
    (@copies partial: $src:expr, $dyn_len:expr => $dst:expr ; $($rest:tt)*) => {
        ($dst)[..$dyn_len].copy_from_slice(&$src[..$dyn_len]);
        handle_match!(@copies $($rest)*);
    };

    // Scalar assignment: scalar: value => dst;
    (@copies scalar: $value:expr => $dst:expr ; $($rest:tt)*) => {
        ($dst)[0] = $value;
        handle_match!(@copies $($rest)*);
    };

    (@copies) => {};
}
