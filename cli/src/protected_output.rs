//! Atomic publication of one output file, with owner-only permissions on Unix.
//! Stage all files before publication; publish private keys only after verifying
//! the winner. Multiple independent paths are not an atomic filesystem group.

use rand::{RngCore, rngs::OsRng};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Resolve and preflight paths before a potentially expensive search. Publication
/// still uses an atomic no-clobber operation to handle races after this check.
pub fn validate_outputs(outputs: &[&Path], inputs: &[&Path], force: bool) -> io::Result<()> {
    fn resolve(path: &Path) -> io::Result<PathBuf> {
        let name = path.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "output must name a file")
        })?;
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        Ok(parent.canonicalize()?.join(name))
    }
    let mut input_paths = Vec::new();
    for path in inputs {
        input_paths.push(path.canonicalize()?);
        input_paths.push(resolve(path)?);
    }
    let mut resolved = Vec::new();
    for output in outputs {
        let output = resolve(output)?;
        if resolved.contains(&output) || input_paths.contains(&output) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "output paths must be distinct and must not replace inputs",
            ));
        }
        match fs::symlink_metadata(&output) {
            Ok(metadata) => {
                if !force {
                    return Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "output exists; use --force to replace it",
                    ));
                }
                if metadata.is_dir() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "output cannot replace a directory",
                    ));
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        resolved.push(output);
    }
    Ok(())
}

/// Staged files contain no path derived from secret material and are removed
/// on ordinary error/unwind. An abrupt process kill can leave a mode-0600 temp
/// file; it never leaves a partially written destination key.
pub struct StagedOutput {
    temporary: PathBuf,
    destination: PathBuf,
    force: bool,
}

impl StagedOutput {
    pub fn new(destination: &Path, contents: &[u8], force: bool) -> io::Result<Self> {
        let name = destination.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "output must name a file")
        })?;
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        // Resolve the parent once so relative path changes cannot redirect commit.
        let parent = parent.canonicalize()?;
        let destination = parent.join(name);
        if !force {
            match fs::symlink_metadata(&destination) {
                Ok(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "output exists; use --force to replace it",
                    ));
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
        for _ in 0..64 {
            let mut random = [0; 16];
            OsRng
                .try_fill_bytes(&mut random)
                .map_err(|_| io::Error::other("OS entropy unavailable for output staging"))?;
            let temporary = parent.join(format!(".vanity-output-{}.tmp", hex::encode(random)));
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = match options.open(&temporary) {
                Ok(file) => file,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            };
            let staged = Self {
                temporary,
                destination,
                force,
            };
            file.write_all(contents)?;
            file.sync_all()?;
            drop(file);
            return Ok(staged);
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not reserve a staging file",
        ))
    }

    /// Commit exactly once. No-clobber publication uses an atomic hard link in
    /// the same directory, so a racing writer cannot be silently overwritten.
    /// A directory-sync failure is reported even though publication may already
    /// have succeeded; callers must not retry with a different winner.
    pub fn publish(self) -> io::Result<()> {
        if self.force {
            fs::rename(&self.temporary, &self.destination)?;
        } else {
            fs::hard_link(&self.temporary, &self.destination)?;
            fs::remove_file(&self.temporary)?;
        }
        #[cfg(unix)]
        File::open(self.destination.parent().expect("resolved output parent"))?.sync_all()?;
        Ok(())
    }
}

impl Drop for StagedOutput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.temporary);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);
    impl TestDirectory {
        fn new() -> Self {
            let mut random = [0; 16];
            OsRng.fill_bytes(&mut random);
            let path =
                std::env::temp_dir().join(format!("vanity-output-test-{}", hex::encode(random)));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn preflight_rejects_input_replacement_and_duplicate_paths() {
        let dir = TestDirectory::new();
        let input = dir.0.join("input");
        let output = dir.0.join("output");
        fs::write(&input, b"public input test bytes").unwrap();
        assert!(validate_outputs(&[&input], &[&input], true).is_err());
        assert!(validate_outputs(&[&output, &dir.0.join("./output")], &[&input], true).is_err());
        assert!(validate_outputs(&[&output], &[&input], false).is_ok());
        #[cfg(unix)]
        {
            let alias = dir.0.join("alias");
            std::os::unix::fs::symlink(&input, &alias).unwrap();
            assert!(validate_outputs(&[&alias], &[&alias], true).is_err());
            assert!(validate_outputs(&[&input], &[&alias], true).is_err());
        }
    }

    #[test]
    fn publish_and_permissions() {
        let dir = TestDirectory::new();
        let path = dir.0.join("key.pem");
        let staged = StagedOutput::new(&path, b"synthetic test bytes", false).unwrap();
        assert!(!path.exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&staged.temporary)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        staged.publish().unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"synthetic test bytes");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
    }

    #[test]
    fn racing_output_is_not_overwritten() {
        let dir = TestDirectory::new();
        let path = dir.0.join("key.pem");
        let first = StagedOutput::new(&path, b"first", false).unwrap();
        let second = StagedOutput::new(&path, b"second", false).unwrap();
        first.publish().unwrap();
        assert_eq!(
            second.publish().unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(fs::read(&path).unwrap(), b"first");
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
        assert!(StagedOutput::new(&path, b"third", false).is_err());
    }

    #[test]
    fn force_replaces_and_abandoned_stage_is_removed() {
        let dir = TestDirectory::new();
        let path = dir.0.join("key.pem");
        fs::write(&path, b"old").unwrap();
        let staged = StagedOutput::new(&path, b"replacement", true).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"old");
        drop(staged);
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
        StagedOutput::new(&path, b"replacement", true)
            .unwrap()
            .publish()
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"replacement");
    }

    #[cfg(unix)]
    #[test]
    fn destination_symlinks_are_not_followed() {
        use std::os::unix::fs::symlink;
        let dir = TestDirectory::new();
        let target = dir.0.join("target");
        let output = dir.0.join("key.pem");
        fs::write(&target, b"unchanged").unwrap();
        symlink(&target, &output).unwrap();
        assert!(StagedOutput::new(&output, b"new", false).is_err());
        StagedOutput::new(&output, b"new", true)
            .unwrap()
            .publish()
            .unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"unchanged");
        assert_eq!(fs::read(&output).unwrap(), b"new");
        assert!(
            !fs::symlink_metadata(output)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }
}
