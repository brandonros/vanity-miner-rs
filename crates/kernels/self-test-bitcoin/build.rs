#[cfg(feature = "cuda")]
#[path = "../build_support.rs"]
mod build_support;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    #[cfg(feature = "cuda")]
    build_support::build("self_test_bitcoin", true);
}
