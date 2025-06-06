use std::env;
use std::path;

use cuda_builder::CudaBuilder;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernel");
    println!("cargo:rerun-if-changed=ed25519");

    unsafe {
        env::set_var("LLVM_CONFIG", "llvm-config-7");
        env::set_var("LLVM_LINK_STATIC", "1");
        env::set_var("RUST_LOG", "info");

        let cuda_lib_path = "/usr/local/cuda/nvvm/lib64/";
        match env::var("LD_LIBRARY_PATH") {
            Ok(existing_path) => {
                env::set_var("LD_LIBRARY_PATH", format!("{}:{}", existing_path, cuda_lib_path));
            }
            Err(_) => {
                env::set_var("LD_LIBRARY_PATH", cuda_lib_path);
            }
        }
    }

    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    CudaBuilder::new("kernel")
        .copy_to(out_path.join("kernel.ptx"))
        .build()
        .unwrap();
}
