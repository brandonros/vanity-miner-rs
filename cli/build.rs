fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "gpu")]
    build_gpu();

    #[cfg(feature = "self_test_support")]
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
    let production_features = production.join(",");
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

    // Keep each libNVVM invocation bounded to one self-test mode. Production
    // kernels retain their existing feature-selected build.
    let build = |name: &str, features: &str| {
        let mut args = kernel_args.clone();
        args.extend(["--features".to_owned(), features.to_owned()]);
        let ptx = out_path.join(format!("{name}.ptx"));
        CudaBuilder::new(&kernels_dir)
            .arch(arch)
            .build_args(&args)
            .copy_to(&ptx)
            .final_module_path(out_path.join("final-module.ll"))
            .emit_llvm_ir(true)
            .build()
            .unwrap();
        ptx
    };
    // OUT_DIR is <profile>/build/<package>/out. Publish standalone artifacts
    // beside the host binary as well as embedding the build-specific copies.
    let artifacts = out_path.ancestors().nth(3).unwrap().join("ptx");
    std::fs::create_dir_all(&artifacts).unwrap();
    if cfg!(feature = "self_test_support") {
        let mut embedded = String::from(
            "fn embedded_self_test_ptx(name: &str) -> Option<&'static str> { match name {\n",
        );
        for &name in &self_tests {
            let ptx = build(name, name);
            std::fs::copy(&ptx, artifacts.join(format!("{name}.ptx"))).unwrap();
            embedded.push_str(&format!(
                "{name:?} => Some(include_str!({:?})),\n",
                ptx.to_str().unwrap()
            ));
        }
        embedded.push_str("_ => None, } }\n");
        std::fs::write(out_path.join("self_test_ptx.rs"), embedded).unwrap();
    }
    let ptx_path = if production_features.is_empty() && !self_tests.is_empty() {
        // Use a real test module for the default embedding in self-test-only builds.
        out_path.join(format!("{}.ptx", self_tests[0]))
    } else {
        build("kernels", &production_features)
    };
    if !production_features.is_empty() {
        std::fs::copy(&ptx_path, artifacts.join("kernels.ptx")).unwrap();
    }
    println!("cargo:rustc-env=KERNELS_PTX_PATH={}", ptx_path.display());
}

#[cfg(feature = "self_test_support")]
fn export_self_test_names() {
    use std::{env, fs, path::PathBuf};
    let directory = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../kernels/src");
    println!("cargo::rerun-if-changed={}", directory.display());
    let mut names = std::collections::BTreeMap::new();
    for source in fs::read_dir(directory).unwrap() {
        let source = source.unwrap().path();
        if !source
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("self_test_")
        {
            continue;
        }
        let mut entry = None;
        for line in fs::read_to_string(source).unwrap().lines() {
            if let Some(tail) = line.trim().strip_prefix(r#"pub unsafe extern "C" fn "#) {
                entry = Some(tail.split('(').next().unwrap().to_owned());
            }
            if let Some(tail) = line.trim().strip_prefix("results[") {
                let slot: usize = tail.split(']').next().unwrap().parse().unwrap();
                let name = entry.as_ref().expect("slot without kernel entry");
                assert!(
                    names.insert(slot, name.clone()).is_none(),
                    "duplicate slot {slot}"
                );
            }
        }
    }
    let count = names.len();
    assert!(count > 0);
    assert_eq!(
        names.keys().copied().collect::<Vec<_>>(),
        (0..count).collect::<Vec<_>>()
    );
    let enabled: Vec<bool> = names
        .values()
        .map(|kernel| {
            let feature = kernel.strip_prefix("kernel_").unwrap().to_ascii_uppercase();
            env::var_os(format!("CARGO_FEATURE_{feature}")).is_some()
        })
        .collect();
    let text = format!(
        "const SELF_TEST_ENTRIES: [&str; {count}] = {:?};\nconst SELF_TEST_ENABLED: [bool; {count}] = {enabled:?};",
        names.values().collect::<Vec<_>>()
    );
    fs::write(
        PathBuf::from(env::var("OUT_DIR").unwrap()).join("self_test_entries.rs"),
        text,
    )
    .unwrap();
}
