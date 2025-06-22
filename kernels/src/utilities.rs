unsafe extern "C" {
    fn __nvvm_thread_idx_x() -> u32;
    fn __nvvm_thread_idx_y() -> u32;
    fn __nvvm_thread_idx_z() -> u32;

    fn __nvvm_block_dim_x() -> u32;
    fn __nvvm_block_dim_y() -> u32;
    fn __nvvm_block_dim_z() -> u32;

    fn __nvvm_block_idx_x() -> u32;
    fn __nvvm_block_idx_y() -> u32;
    fn __nvvm_block_idx_z() -> u32;

    fn __nvvm_grid_dim_x() -> u32;
    fn __nvvm_grid_dim_y() -> u32;
    fn __nvvm_grid_dim_z() -> u32;

    fn __nvvm_warp_size() -> u32;

    fn __nvvm_block_barrier();

    fn __nvvm_grid_fence();
    fn __nvvm_device_fence();
    fn __nvvm_system_fence();
}

pub fn get_thread_idx() -> usize {
    let thread_idx_x = unsafe { __nvvm_thread_idx_x() };
    let thread_idx_y = unsafe { __nvvm_thread_idx_y() };
    let thread_idx_z = unsafe { __nvvm_thread_idx_z() };

    let block_dim_x = unsafe { __nvvm_block_dim_x() };
    let block_dim_y = unsafe { __nvvm_block_dim_y() };
    let block_dim_z = unsafe { __nvvm_block_dim_z() };

    let grid_dim_x = unsafe { __nvvm_grid_dim_x() };
    let grid_dim_y = unsafe { __nvvm_grid_dim_y() };
    let _grid_dim_z = unsafe { __nvvm_grid_dim_z() };

    let block_idx_x = unsafe { __nvvm_block_idx_x() };
    let block_idx_y = unsafe { __nvvm_block_idx_y() };
    let block_idx_z = unsafe { __nvvm_block_idx_z() };

    let block_dim_product = block_dim_x * block_dim_y * block_dim_z;
    let block_id = block_idx_x + block_idx_y * grid_dim_x 
                       + grid_dim_x * grid_dim_y * block_idx_z;

    let thread_idx = block_id * block_dim_product
    + (thread_idx_z * (block_dim_x * block_dim_y))
    + (thread_idx_y * block_dim_x) + thread_idx_x;

    thread_idx as usize
}
