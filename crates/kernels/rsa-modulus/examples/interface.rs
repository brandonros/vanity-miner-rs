#[path = "../../common/candidate_interface.rs"]
mod interface;
fn main() {
    println!(
        "{}",
        interface::interface::<kernel_rsa_modulus_metal::RsaModulus>()
    );
}
