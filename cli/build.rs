fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "gpu")]
    build_gpu();

    #[cfg(feature = "cumetal")]
    pin_cumetal();
}

#[cfg(feature = "cumetal")]
fn pin_cumetal() {
    let lock_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../flake.lock");
    println!("cargo::rerun-if-changed={}", lock_path.display());
    println!("cargo::rerun-if-env-changed=VANITY_CUMETAL_ROOT");
    let lock: serde_json::Value =
        serde_json::from_slice(&std::fs::read(lock_path).expect("read flake.lock"))
            .expect("parse flake.lock");
    let root = lock["root"].as_str().expect("flake root");
    let input = lock["nodes"][root]["inputs"]["cumetal"]
        .as_str()
        .expect("CuMetal input must be pinned in flake.lock");
    let pinned = &lock["nodes"][input]["locked"];
    let revision = pinned["rev"].as_str().expect("CuMetal revision");
    assert!(revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()));
    println!("cargo::rustc-env=VANITY_CUMETAL_REVISION={revision}");
    println!(
        "cargo::rustc-env=VANITY_CUMETAL_SOURCE={}",
        pinned["url"].as_str().expect("CuMetal Git source URL")
    );
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
        ("rsa-modulus", cfg!(feature = "rsa-modulus")),
        ("rsa-pss", cfg!(feature = "rsa-pss")),
        ("p256-public-key", cfg!(feature = "p256-public-key")),
        ("p256-signature", cfg!(feature = "p256-signature")),
        ("self_test_solana", cfg!(feature = "self_test_solana")),
        ("self_test_bitcoin", cfg!(feature = "self_test_bitcoin")),
        ("self_test_ethereum", cfg!(feature = "self_test_ethereum")),
        ("self_test_shallenge", cfg!(feature = "self_test_shallenge")),
        (
            "self_test_p256_public_key",
            cfg!(feature = "self_test_p256_public_key"),
        ),
        (
            "self_test_p256_signature",
            cfg!(feature = "self_test_p256_signature"),
        ),
        ("self_test_rsa_pss", cfg!(feature = "self_test_rsa_pss")),
        (
            "self_test_rsa_modulus",
            cfg!(feature = "self_test_rsa_modulus"),
        ),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect::<Vec<_>>();
    let (self_tests, production): (Vec<_>, Vec<_>) = kernel_features
        .into_iter()
        .partition(|name| name.starts_with("self_test_"));
    let mut kernel_args = vec!["--no-default-features".to_owned(), "--locked".to_owned()];
    // Legacy libnvvm rejects vector bswap emitted while optimizing HMAC at O3.
    // Keep the workaround in the nested kernel build, preserving host and
    // LLVM 21 optimization. O1 avoids the legacy vectorization pipeline.
    if !cfg!(feature = "llvm21")
        && (cfg!(feature = "crypto-cli") || cfg!(feature = "self_test_support"))
    {
        kernel_args.extend([
            "--config".to_owned(),
            "profile.release.opt-level=1".to_owned(),
        ]);
    }

    // The modern NVVM dialect requires a Blackwell-or-later target.
    let arch = if cfg!(feature = "llvm21") {
        NvvmArch::Compute100
    } else {
        NvvmArch::Compute89
    };

    // Every selected feature gets its own libNVVM invocation and PTX module.
    let build = |name: &str, features: &str| {
        let mut args = kernel_args.clone();
        args.extend(["--features".to_owned(), features.to_owned()]);
        let ptx = out_path.join(format!("{name}.ptx"));
        CudaBuilder::new(&kernels_dir)
            .arch(arch)
            .build_args(&args)
            .copy_to(&ptx)
            .final_module_path(out_path.join(format!("{name}.ll")))
            .emit_llvm_ir(true)
            .build()
            .unwrap();
        ptx
    };
    // OUT_DIR is <profile>/build/<package>/out. Publish standalone artifacts
    // beside the host binary as well as embedding the build-specific copies.
    let artifacts = out_path.ancestors().nth(3).unwrap().join("ptx");
    std::fs::create_dir_all(&artifacts).unwrap();
    // Retire the former combined artifact so it cannot be mistaken for current output.
    let combined = artifacts.join("kernels.ptx");
    if combined.exists() {
        std::fs::remove_file(combined).unwrap();
    }
    let mut embedded =
        String::from("fn embedded_ptx(name: &str) -> Option<&'static str> { match name {\n");
    for feature in production.iter().chain(self_tests.iter()) {
        let name = feature.replace('-', "_");
        let ptx = build(&name, feature);
        std::fs::copy(&ptx, artifacts.join(format!("{name}.ptx"))).unwrap();
        embedded.push_str(&format!(
            "{name:?} => Some(include_str!({:?})),\n",
            ptx.to_str().unwrap()
        ));
    }
    embedded.push_str("_ => None, } }\n");
    std::fs::write(out_path.join("kernel_ptx.rs"), embedded).unwrap();
}
