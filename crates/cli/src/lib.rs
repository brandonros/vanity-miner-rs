//! Mode-owned CLI behavior and shared execution infrastructure.
#[cfg(all(feature = "metal", not(target_os = "macos")))]
compile_error!("The metal backend requires macOS");

mod application;
mod args;
pub mod modes;
pub mod runner;
pub use application::run_cli;
