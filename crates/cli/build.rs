fn main() {
    // On Windows, nanorand's entropy uses SystemFunction036 (RtlGenRandom) from advapi32.
    // Explicitly link it so the MSVC linker resolves the symbol (avoids LNK2019 when
    // mixing CRTs or with certain link orders).
    #[cfg(all(feature = "gpu", target_os = "windows"))]
    println!("cargo:rustc-link-lib=advapi32");

    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "cumetal")]
    pin_cumetal();
}

#[cfg(feature = "cumetal")]
fn pin_cumetal() {
    let lock_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../flake.lock");
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
