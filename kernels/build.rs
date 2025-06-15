use std::env;
use std::path;

use cuda_builder::CudaBuilder;

fn main() {
    env_logger::init();
    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    CudaBuilder::new("./")
        .copy_to(out_path.join("kernels.ptx"))
        .build()
        .unwrap();
}
