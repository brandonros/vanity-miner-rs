#[path = "../../common/candidate_interface.rs"]
mod interface;
fn main() {
    println!(
        "{}",
        interface::interface::<kernel_p256_signature_metal::P256Signature>()
    );
}
