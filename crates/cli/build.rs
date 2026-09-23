//! With `gpu`, compiles the kernel crate of each enabled mode for the built-in
//! nvptx64-nvidia-cuda target. runner/cuda/module.rs embeds the PTX.
use std::{env, fs, path::PathBuf, process::Command};

/// PTX modules, named after the features that enable them.
const MODULES: [&str; 14] = [
    "solana",
    "bitcoin",
    "ethereum",
    "shallenge",
    "p256_public_key",
    "p256_signature",
    "rsa_pss",
    "self_test_solana",
    "self_test_bitcoin",
    "self_test_ethereum",
    "self_test_shallenge",
    "self_test_p256_public_key",
    "self_test_p256_signature",
    "self_test_rsa_pss",
];

/// Oldest GPU architecture the PTX supports. The driver compiles it for newer GPUs.
const ARCH: &str = "sm_75";

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    let modules: Vec<&str> = MODULES
        .into_iter()
        .filter(|module| env::var_os(format!("CARGO_FEATURE_{}", module.to_uppercase())).is_some())
        .collect();
    if env::var_os("CARGO_FEATURE_GPU").is_none() || modules.is_empty() {
        return;
    }
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    for input in ["Cargo.toml", "Cargo.lock", "crates/logic", "crates/kernels"] {
        println!("cargo::rerun-if-changed={}", root.join(input).display());
    }

    // OUT_DIR is <target>/[<triple>/]<profile>/build/<package>/out. The kernels
    // need another target directory: this build holds the lock on its own.
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let target_dir = out.ancestors().nth(4).unwrap().join("nvptx");
    let mut cargo = Command::new(env::var_os("CARGO").unwrap());
    cargo
        .current_dir(&root)
        .args(["build", "--release", "--locked", "--target", "nvptx64-nvidia-cuda"])
        .args(["-Zbuild-std=core,alloc", "--target-dir"])
        .arg(&target_dir)
        // Replace the host's flags, and do not run clippy on the kernels.
        .env("CARGO_ENCODED_RUSTFLAGS", format!("-Ctarget-cpu={ARCH}"))
        .env_remove("RUSTC_WORKSPACE_WRAPPER");
    for module in &modules {
        cargo.args(["-p", &format!("kernel-{}", module.replace('_', "-"))]);
    }
    let status = cargo.status().expect("run cargo");
    assert!(
        status.success(),
        "kernel build failed; it needs the toolchain from rust-toolchain.toml"
    );
    let ptx = target_dir.join("nvptx64-nvidia-cuda/release");
    for module in modules {
        let file = format!("{module}.ptx");
        fs::copy(ptx.join(&file), out.join(&file)).unwrap();
    }
}
