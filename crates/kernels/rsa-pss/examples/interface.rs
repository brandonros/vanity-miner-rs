#[path = "../../common/candidate_interface.rs"]
mod interface;
fn main() {
    println!("{}", interface::interface::<kernel_rsa_pss_metal::RsaPss>());
}
