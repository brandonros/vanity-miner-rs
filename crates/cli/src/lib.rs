//! Mode-owned CLI behavior and shared execution infrastructure.
mod application;
mod args;
pub mod modes;
pub mod runner;
pub use application::run_cli;
