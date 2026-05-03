fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "gpu")]
    build_gpu();
}

#[cfg(feature = "gpu")]
fn build_gpu() {
    use std::env;
    use std::path::PathBuf;

    use cuda_builder::CudaBuilder;

    // On Windows, nanorand's entropy uses SystemFunction036 (RtlGenRandom) from advapi32.
    // Explicitly link it so the MSVC linker resolves the symbol (avoids LNK2019 when
    // mixing CRTs or with certain link orders).
    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-lib=advapi32");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.parent().unwrap().join("kernels");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo::rerun-if-changed={}", kernels_dir.display());
    println!("cargo::rerun-if-env-changed=RUST_CUDA_DUMP_FINAL_MODULE");
    println!("cargo::rerun-if-env-changed=RUST_CUDA_EMIT_LLVM_IR");

    let dump_final_module = env::var_os("RUST_CUDA_DUMP_FINAL_MODULE").is_some();
    let emit_llvm_ir = env::var_os("RUST_CUDA_EMIT_LLVM_IR").is_some();

    let ptx_path = out_path.join("kernels.ptx");
    let mut builder = CudaBuilder::new(&kernels_dir);
    builder = builder.copy_to(&ptx_path);

    if dump_final_module {
        builder = builder.final_module_path(out_path.join("final-module.ll"));
    }

    if emit_llvm_ir {
        builder = builder.emit_llvm_ir(true);
    }

    builder.build().unwrap();

    println!("cargo:rustc-env=KERNELS_PTX_PATH={}", ptx_path.display());
}
