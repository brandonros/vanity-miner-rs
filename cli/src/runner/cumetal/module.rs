use super::{
    CumetalRunner, Error,
    driver::{Driver, Module},
};
use std::{
    path::{Path, PathBuf},
    rc::Rc,
};

impl CumetalRunner {
    pub(crate) fn module(&self, driver: &Rc<Driver>, entry: &str) -> Result<Module, Error> {
        let mut temporary = None;
        let path = if let Some(directory) = &self.options.module_dir {
            directory.join(format!("{entry}.metal"))
        } else {
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
            let compiler = resolve_program(&self.options.cumetalc)?;

            temporary = Some(TemporaryDirectory::new()?);
            let output = temporary.as_ref().unwrap().0.join(format!("{entry}.metal"));
            eprintln!("Compiling {entry} from {}", input.display());
            let status = std::process::Command::new(&compiler)
                .arg(input)
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
            output
        };
        if !path.with_extension("metal.cumetal-abi").is_file() {
            return Err(format!("Missing ABI sidecar for {}", path.display()).into());
        }
        let mut module = driver.module(&path, entry)?;
        module.temporary = temporary;
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

fn resolve_program(path: &Path) -> Result<PathBuf, Error> {
    if path.components().count() > 1 || path.is_file() {
        return Ok(path.canonicalize()?);
    }
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let candidate = directory.join(path);
        if candidate.is_file() {
            return Ok(candidate.canonicalize()?);
        }
    }
    Err(format!("Cannot find {} on PATH", path.display()).into())
}
