pub fn get_thread_idx() -> usize {
    cuda_std::thread::index() as usize
}
