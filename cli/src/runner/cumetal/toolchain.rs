//! Verify the compiler/runtime package against the consumer's locked dependency.
use super::Error;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildManifest {
    schema: u32,
    revision: String,
    source: String,
    source_path: PathBuf,
    compiler_sha256: String,
    runtime_sha256: String,
}

pub(super) struct Toolchain {
    pub compiler: PathBuf,
    pub runtime: PathBuf,
}

impl Toolchain {
    pub fn load(root: &Path) -> Result<Self, Error> {
        if !root.is_absolute() {
            return Err("--cumetal-root must be an absolute package path".into());
        }
        let root = root.canonicalize()?;
        let manifest_path = root.join("share/cumetal/build.json");
        let manifest: BuildManifest = serde_json::from_slice(
            &std::fs::read(&manifest_path)
                .map_err(|e| format!("Cannot read {}: {e}", manifest_path.display()))?,
        )?;
        if manifest.schema != 1
            || manifest.revision != env!("VANITY_CUMETAL_REVISION")
            || manifest.source != env!("VANITY_CUMETAL_SOURCE")
        {
            return Err(format!(
                "CuMetal package does not match flake.lock: expected {}@{}, found {}@{} (schema {}). Rebuild with `nix develop .#cumetal`",
                env!("VANITY_CUMETAL_SOURCE"),
                env!("VANITY_CUMETAL_REVISION"),
                manifest.source,
                manifest.revision,
                manifest.schema,
            )
            .into());
        }
        let compiler = root.join("bin/cumetalc").canonicalize()?;
        let runtime = root.join("lib/libcumetal.dylib").canonicalize()?;
        for (path, expected) in [
            (&compiler, &manifest.compiler_sha256),
            (&runtime, &manifest.runtime_sha256),
        ] {
            let actual = fingerprint(&std::fs::read(path)?);
            if actual != *expected {
                return Err(format!(
                    "CuMetal artifact does not match its build manifest: {}",
                    path.display()
                )
                .into());
            }
        }
        eprintln!("CuMetal source: {}@{}", manifest.source, manifest.revision);
        eprintln!("CuMetal source path: {}", manifest.source_path.display());
        eprintln!(
            "CuMetal compiler: {} sha256={}",
            compiler.display(),
            manifest.compiler_sha256
        );
        eprintln!(
            "CuMetal runtime: {} sha256={}",
            runtime.display(),
            manifest.runtime_sha256
        );
        Ok(Self { compiler, runtime })
    }
}

pub(super) fn fingerprint(bytes: &[u8]) -> String {
    hex::encode(logic::crypto::sha256::sha256_from_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Package(PathBuf);
    impl Package {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "vanity-cumetal-package-{:016x}",
                rand::random::<u64>()
            ));
            for directory in ["bin", "lib", "share/cumetal"] {
                std::fs::create_dir_all(root.join(directory)).unwrap();
            }
            std::fs::write(root.join("bin/cumetalc"), b"compiler").unwrap();
            std::fs::write(root.join("lib/libcumetal.dylib"), b"runtime").unwrap();
            let package = Self(root);
            package.manifest(env!("VANITY_CUMETAL_REVISION"));
            package
        }
        fn manifest(&self, revision: &str) {
            let value = serde_json::json!({
                "schema": 1,
                "revision": revision,
                "source": env!("VANITY_CUMETAL_SOURCE"),
                "source_path": "/immutable/source",
                "compiler_sha256": fingerprint(b"compiler"),
                "runtime_sha256": fingerprint(b"runtime"),
            });
            std::fs::write(self.0.join("share/cumetal/build.json"), value.to_string()).unwrap();
        }
    }
    impl Drop for Package {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn rejects_another_revision_even_with_matching_artifact_hashes() {
        let package = Package::new();
        package.manifest(&"0".repeat(40));
        let error = Toolchain::load(&package.0).err().unwrap().to_string();
        assert!(error.contains("does not match flake.lock"));
    }

    #[test]
    fn rejects_a_replaced_compiler_or_runtime_before_loading() {
        for file in ["bin/cumetalc", "lib/libcumetal.dylib"] {
            let package = Package::new();
            std::fs::write(package.0.join(file), b"another build").unwrap();
            let error = Toolchain::load(&package.0).err().unwrap().to_string();
            assert!(error.contains("does not match its build manifest"));
        }
    }

    #[test]
    fn accepts_matching_pair_and_rejects_missing_manifest() {
        let package = Package::new();
        let toolchain = Toolchain::load(&package.0).unwrap();
        assert_eq!(
            toolchain.compiler,
            package.0.join("bin/cumetalc").canonicalize().unwrap()
        );
        assert_eq!(
            toolchain.runtime,
            package
                .0
                .join("lib/libcumetal.dylib")
                .canonicalize()
                .unwrap()
        );
        std::fs::remove_file(package.0.join("share/cumetal/build.json")).unwrap();
        assert!(Toolchain::load(&package.0).is_err());
        assert!(Toolchain::load(Path::new("build-nix")).is_err());
    }
}
