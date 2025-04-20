use std::env;
use std::path;

use cuda_builder::CudaBuilder;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernel");

    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    CudaBuilder::new("kernel")
        .copy_to(out_path.join("kernel.ptx"))
        .build()
        .unwrap();
}
