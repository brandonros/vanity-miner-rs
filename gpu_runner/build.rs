use std::env;
use std::path;

use cuda_builder::CudaBuilder;
use cuda_builder::NvvmArch;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../kernels");
    println!("cargo:rerun-if-changed=../common");
    println!("cargo:rerun-if-changed=../logic");
    
    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    CudaBuilder::new("../kernels")
        .copy_to(out_path.join("kernels.ptx"))
        .arch(NvvmArch::Compute70)
        .build()
        .unwrap();
}
