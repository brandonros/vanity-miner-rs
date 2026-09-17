//! Mode-owned CLI behavior and shared execution infrastructure.
#[cfg(all(feature = "gpu", feature = "cumetal"))]
compile_error!("Select either gpu (NVIDIA CUDA) or cumetal, not both");
mod application;
mod args;
pub mod modes;
pub mod runner;
pub use application::run_cli;
