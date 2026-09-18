use llvm_metal_runtime::Kernel;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};
/// Validate bundle hashes and exact application bindings before creating a pipeline.
pub(crate) fn load_artifact(
    directory: &Path,
    interface_json: &str,
) -> Result<(Kernel, Duration), String> {
    let read = |name| fs::read(directory.join(name)).map_err(|e| format!("{name}: {e}"));
    let manifest: serde_json::Value =
        serde_json::from_slice(&read("kernel.build.json")?).map_err(|e| e.to_string())?;
    if manifest["schema"] != 1 {
        return Err("unsupported Metal artifact schema".into());
    }
    for name in ["kernel.metallib", "kernel.bindings.json"] {
        if manifest["artifacts"][name].as_str()
            != Some(hex::encode(Sha256::digest(read(name)?)).as_str())
        {
            return Err(format!("Metal artifact hash mismatch: {name}"));
        }
    }
    let bindings: llvm_metal_abi::MetalBindings =
        serde_json::from_slice(&read("kernel.bindings.json")?).map_err(|e| e.to_string())?;
    let interface: llvm_metal_abi::KernelInterface =
        serde_json::from_str(interface_json).map_err(|e| e.to_string())?;
    if serde_json::to_value(&bindings).unwrap()
        != serde_json::to_value(interface.validate()?).unwrap()
    {
        return Err("Metal kernel bindings do not match the application ABI".into());
    }
    let start = Instant::now();
    let kernel = Kernel::load(&directory.join("kernel.metallib"), &bindings)?;
    let load_time = start.elapsed();
    let stages = kernel.load_timings();
    eprintln!(
        "Metal device: {}; entry {}; library load {:.3} ms; pipeline creation {:.3} ms; total load {:.3} ms",
        kernel.device_name(),
        bindings.entry,
        stages.library.as_secs_f64() * 1000.,
        stages.pipeline.as_secs_f64() * 1000.,
        load_time.as_secs_f64() * 1000.
    );
    Ok((kernel, load_time))
}
