//! Publish source and ABI together; failed or concurrent builds cannot expose half a module.
use super::Error;
use std::path::{Path, PathBuf};

struct TemporaryDirectory(PathBuf);
impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn complete(path: &Path) -> bool {
    path.is_file() && path.with_extension("metal.cumetal-abi").is_file()
}
pub(super) fn module(
    directory: &Path,
    entry: &str,
    build: impl FnOnce(&Path) -> Result<(), Error>,
) -> Result<PathBuf, Error> {
    let output = directory.join(format!("{entry}.metal"));
    if complete(&output) {
        return Ok(output);
    }
    let parent = directory.parent().ok_or("cache directory has no parent")?;
    std::fs::create_dir_all(parent)?;
    let temporary = loop {
        let path = parent.join(format!(
            ".building-{}-{:016x}",
            std::process::id(),
            rand::random::<u64>()
        ));
        match std::fs::create_dir(&path) {
            Ok(()) => break TemporaryDirectory(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    };
    let candidate = temporary.0.join(format!("{entry}.metal"));
    build(&candidate)?;
    if !complete(&candidate) {
        return Err("CuMetal compilation did not produce both Metal source and ABI sidecar".into());
    }
    if let Err(error) = std::fs::rename(&temporary.0, directory) {
        // Another process may have published the same content-addressed entry.
        if !complete(&output) {
            return Err(format!(
                "Cannot publish module cache {}: {error}",
                directory.display()
            )
            .into());
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn root() -> TemporaryDirectory {
        let path =
            std::env::temp_dir().join(format!("vanity-cache-test-{:016x}", rand::random::<u64>()));
        std::fs::create_dir(&path).unwrap();
        TemporaryDirectory(path)
    }
    fn write(path: &Path, text: &str) -> Result<(), Error> {
        std::fs::write(path, text)?;
        std::fs::write(path.with_extension("metal.cumetal-abi"), "CUMETAL_ABI_V2\n")?;
        Ok(())
    }
    #[test]
    fn failed_build_cannot_poison_retry() {
        let root = root();
        let cache = root.0.join("hash");
        assert!(
            module(&cache, "kernel", |path| {
                std::fs::write(path, "partial")?;
                Err("compiler failed".into())
            })
            .is_err()
        );
        assert!(!cache.exists());
        let output = module(&cache, "kernel", |path| write(path, "complete")).unwrap();
        assert_eq!(std::fs::read_to_string(output).unwrap(), "complete");
        module(&cache, "kernel", |_| panic!("cache hit must not rebuild")).unwrap();
    }
    #[test]
    fn missing_sidecar_is_not_published() {
        let root = root();
        let cache = root.0.join("hash");
        assert!(
            module(&cache, "kernel", |path| {
                std::fs::write(path, "source")?;
                Ok(())
            })
            .is_err()
        );
        assert!(!cache.exists());
    }
    #[test]
    fn competing_publisher_wins_without_partial_overwrite() {
        let root = root();
        let cache = root.0.join("hash");
        let output = module(&cache, "kernel", |candidate| {
            std::fs::create_dir(&cache)?;
            write(&cache.join("kernel.metal"), "winner")?;
            write(candidate, "loser")
        })
        .unwrap();
        assert_eq!(std::fs::read_to_string(output).unwrap(), "winner");
        assert_eq!(std::fs::read_dir(&root.0).unwrap().count(), 1);
    }
}
