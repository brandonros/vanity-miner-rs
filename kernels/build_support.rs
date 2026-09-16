//! Shared implementation included by each kernel's own build script.
use std::{env, fs, path::PathBuf};

use cuda_builder::{CudaBuilder, NvvmArch};

pub fn build(module: &str, legacy_low_opt: bool) {
    let package = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = package.parent().unwrap().parent().unwrap();
    let device = package.join("device");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    // Watch this mode and its shared inputs, never sibling kernel directories.
    for input in [
        "kernels/build_support.rs",
        "logic",
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
    ] {
        println!("cargo::rerun-if-changed={}", root.join(input).display());
    }
    println!(
        "cargo::rerun-if-changed={}",
        package.join("Cargo.toml").display()
    );
    println!("cargo::rerun-if-changed={}", device.display());
    if !module.starts_with("self_test_")
        && !matches!(module, "rsa_modulus" | "repro_nonce_sequence")
    {
        println!(
            "cargo::rerun-if-changed={}",
            root.join("kernels/common/match_handler.rs").display()
        );
    }

    let mut args = vec!["--no-default-features".to_owned(), "--locked".to_owned()];
    // Legacy libNVVM rejects vector bswap from optimized HMAC. Preserve the
    // workaround for crypto and self-test builds without lowering host opt levels.
    if !cfg!(feature = "llvm21") && legacy_low_opt {
        args.extend([
            "--config".to_owned(),
            "profile.release.opt-level=1".to_owned(),
        ]);
    }
    let arch = if cfg!(feature = "llvm21") {
        NvvmArch::Compute100
    } else {
        NvvmArch::Compute89
    };

    // OUT_DIR is <profile>/build/<package>/out. Each package publishes only its
    // own files, so independent builds cannot remove another mode's artifacts.
    let artifacts = out.ancestors().nth(3).unwrap().join("ptx");
    fs::create_dir_all(&artifacts).unwrap();
    let ptx = out.join(format!("{module}.ptx"));
    CudaBuilder::new(&device)
        .arch(arch)
        .build_args(&args)
        .copy_to(&ptx)
        .final_module_path(out.join(format!("{module}.ll")))
        .emit_llvm_ir(true)
        .build()
        .unwrap_or_else(|error| panic!("failed to build {module}: {error:?}"));
    fs::copy(&ptx, artifacts.join(format!("{module}.ptx"))).unwrap();
}
