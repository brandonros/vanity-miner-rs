//! Mode-owned CLI behavior and shared execution infrastructure.
#[cfg(all(feature = "metal", any(feature = "gpu", feature = "cumetal")))]
compile_error!("Select only one device backend: metal, gpu or cumetal");
#[cfg(all(feature = "metal", not(target_os = "macos")))]
compile_error!("The metal backend requires macOS");

#[cfg(all(feature = "gpu", feature = "cumetal"))]
compile_error!("Select either gpu (NVIDIA CUDA) or cumetal, not both");
mod application;
mod args;
pub mod modes;
pub mod runner;
pub use application::run_cli;
