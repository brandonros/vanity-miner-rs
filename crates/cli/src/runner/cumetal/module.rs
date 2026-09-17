use super::{
    CumetalRunner, Error,
    driver::{Driver, Module},
};
use std::{path::PathBuf, rc::Rc};

impl CumetalRunner {
    pub(crate) fn module(&self, driver: &Rc<Driver>, entry: &str) -> Result<Module, Error> {
        let input = self.options.ptx.as_ref().ok_or("--ptx is required")?;
        let module_path;
        let input = if input.is_dir() {
            #[cfg(feature = "self_test_support")]
            let name = if entry.starts_with("kernel_self_test_") {
                crate::runner::modules::self_test_module(entry)
            } else {
                crate::runner::modules::production_module(entry)
            };
            #[cfg(not(feature = "self_test_support"))]
            let name = crate::runner::modules::production_module(entry);
            module_path = input.join(format!("{name}.ptx"));
            &module_path
        } else {
            input
        };
        let input = input.canonicalize()?;
        let bytes = std::fs::read(&input)?;
        let temporary = TemporaryDirectory::new()?;
        // Compile exactly the bytes whose hash we report, even if an external
        // build replaces the original PTX while translation is running.
        let snapshot = temporary.0.join("input.ptx");
        std::fs::write(&snapshot, &bytes)?;
        let output = temporary.0.join(format!("{entry}.metal"));
        eprintln!(
            "CuMetal PTX: {} sha256={} entry={entry}",
            input.display(),
            super::toolchain::fingerprint(&bytes)
        );
        let status = std::process::Command::new(&self.toolchain.compiler)
            .arg(&snapshot)
            .args([
                "--backend=cumetal-ir",
                "--ptx-strict",
                "--overwrite",
                "--entry",
                entry,
                "--emit=msl",
                "-o",
            ])
            .arg(&output)
            .status()?;
        if !status.success() {
            return Err(format!("CuMetal compilation failed for {entry}: {status}").into());
        }
        if !output.with_extension("metal.cumetal-abi").is_file() {
            return Err(format!("Missing ABI sidecar for {}", output.display()).into());
        }
        let mut module = driver.module(&output, entry)?;
        module.temporary = Some(temporary);
        Ok(module)
    }
}
// CuMetal loads/compiles Metal lazily at launch; retain files until module unload.
pub(super) struct TemporaryDirectory(PathBuf);
impl TemporaryDirectory {
    fn new() -> Result<Self, Error> {
        let path = std::env::temp_dir().join(format!(
            "vanity-cumetal-{}-{:016x}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
