/// Macro to reduce boilerplate in GPU buffer `run_iteration` implementations.
///
/// Handles:
/// 1. Device buffer allocation for prefix/suffix (common to all chains)
/// 2. Device buffer allocation for chain-specific outputs
/// 3. Stream synchronization
/// 4. Conditional copy-back (only if match found)
///
/// The `launch` block has access to these variables:
/// - `prefix_dev`, `prefix_len`, `suffix_dev`, `suffix_len`, `rng_seed`
/// - `found_matches_dev`, `found_thread_idx_dev`
/// - For each field in `output_buffers`: `{field}_dev`
#[macro_export]
macro_rules! impl_run_iteration {
    (
        $self:ident, $kernel:expr, $gpu:expr,
        $prefix:expr, $suffix:expr, $rng_seed:expr,
        output_buffers: [ $( $field:ident ),* $(,)? ],
        launch: $launch_block:block
    ) => {{
        use cust::memory::CopyDestination;
        use cust::util::SliceExt;

        let prefix_len = $prefix.len();
        let suffix_len = $suffix.len();

        // Common input buffers
        let prefix_dev = $prefix.as_dbuf()?;
        let suffix_dev = $suffix.as_dbuf()?;

        // Always-present output buffers
        let found_matches_dev = $self.found_matches_slice.as_dbuf()?;
        let found_thread_idx_dev = $self.found_thread_idx_slice.as_dbuf()?;

        // Chain-specific output buffers
        $(
            paste::paste! { let [<$field _dev>] = $self.$field.as_dbuf()?; }
        )*

        // User-provided kernel launch
        $launch_block

        // Sync and copy back
        $gpu.stream.synchronize()?;
        found_matches_dev.copy_to(&mut $self.found_matches_slice)?;

        // Only copy remaining data if we found a match
        if $self.found_matches_slice[0] != 0 {
            $(
                paste::paste! { [<$field _dev>].copy_to(&mut $self.$field)?; }
            )*
            found_thread_idx_dev.copy_to(&mut $self.found_thread_idx_slice)?;
        }

        Ok(())
    }};
}
