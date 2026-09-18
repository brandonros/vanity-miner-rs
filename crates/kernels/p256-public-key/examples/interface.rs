#[path = "../../common/candidate_interface.rs"]
mod interface;
fn main() { println!("{}", interface::interface::<kernel_p256_public_key_metal::P256Public>()); }
