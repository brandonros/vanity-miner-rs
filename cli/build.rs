fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "gpu")]
    build_gpu();

    #[cfg(all(feature = "cumetal", feature = "self_test"))]
    export_self_test_names();
}

#[cfg(feature = "gpu")]
fn build_gpu() {
    use std::env;
    use std::path::PathBuf;

    use cuda_builder::{CudaBuilder, NvvmArch};

    // On Windows, nanorand's entropy uses SystemFunction036 (RtlGenRandom) from advapi32.
    // Explicitly link it so the MSVC linker resolves the symbol (avoids LNK2019 when
    // mixing CRTs or with certain link orders).
    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-lib=advapi32");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_dir = manifest_dir.parent().unwrap();
    let kernels_dir = workspace_dir.join("kernels");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo::rerun-if-changed={}", kernels_dir.display());
    // The nested kernel workspace also compiles shared logic outside kernels/.
    // Track its sources and the manifests, lockfiles, and toolchain that affect
    // the build so CPU changes cannot leave the embedded PTX stale.
    // kernels/ above includes its own Cargo.toml and Cargo.lock.
    for input in [
        "logic",
        "Cargo.toml",
        "Cargo.lock",
        "cli/Cargo.toml",
        "rust-toolchain.toml",
    ] {
        println!(
            "cargo::rerun-if-changed={}",
            workspace_dir.join(input).display()
        );
    }

    // `kernels` is a separate workspace; Cargo does not forward CLI features
    // into CudaBuilder's nested build automatically.
    let kernel_features = [
        ("solana", cfg!(feature = "solana")),
        ("bitcoin", cfg!(feature = "bitcoin")),
        ("ethereum", cfg!(feature = "ethereum")),
        ("shallenge", cfg!(feature = "shallenge")),
        ("self_test", cfg!(feature = "self_test")),
        ("rsa-modulus", cfg!(feature = "rsa-modulus")),
        ("rsa-pss", cfg!(feature = "rsa-pss")),
        ("p256-public-key", cfg!(feature = "p256-public-key")),
        ("p256-signature", cfg!(feature = "p256-signature")),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect::<Vec<_>>()
    .join(",");
    let mut kernel_args = vec!["--no-default-features".to_owned(), "--locked".to_owned()];
    if !kernel_features.is_empty() {
        kernel_args.extend(["--features".to_owned(), kernel_features]);
    }

    // The modern NVVM dialect requires a Blackwell-or-later target.
    let arch = if cfg!(feature = "llvm21") {
        NvvmArch::Compute100
    } else {
        NvvmArch::Compute89
    };

    let ptx_path = out_path.join("kernels.ptx");
    CudaBuilder::new(&kernels_dir)
        .arch(arch)
        .build_args(&kernel_args)
        .copy_to(&ptx_path)
        .final_module_path(out_path.join("final-module.ll"))
        .emit_llvm_ir(true)
        .build()
        .unwrap();

    println!("cargo:rustc-env=KERNELS_PTX_PATH={}", ptx_path.display());
}

#[cfg(all(feature = "cumetal", feature = "self_test"))]
fn export_self_test_names() {
    use std::{env, fs, path::PathBuf};
    let source = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../kernels/src/self_test.rs");
    println!("cargo::rerun-if-changed={}", source.display());
    let mut names = std::collections::BTreeMap::new();
    let mut entry = None;
    for line in fs::read_to_string(source).unwrap().lines() {
        if let Some(tail) = line.trim().strip_prefix(r#"pub unsafe extern "C" fn "#) {
            entry = Some(tail.split('(').next().unwrap().to_owned());
        }
        if let Some(tail) = line.trim().strip_prefix("results[") {
            let slot: usize = tail.split(']').next().unwrap().parse().unwrap();
            if let Some(name) = entry.take() {
                if name != "kernel_self_test_stub" { assert!(names.insert(slot, name).is_none()); }
            }
        }
    }
    assert_eq!(names.keys().copied().collect::<Vec<_>>(), (0..118).collect::<Vec<_>>());
    let text = format!("const SELF_TEST_ENTRIES: [&str; 118] = {:?};", names.values().collect::<Vec<_>>());
    fs::write(PathBuf::from(env::var("OUT_DIR").unwrap()).join("self_test_entries.rs"), text).unwrap();
}
