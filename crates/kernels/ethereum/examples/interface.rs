#[path = "../../common/candidate_interface.rs"]
mod interface;
fn main() { println!("{}", interface::interface::<kernel_ethereum_metal::Ethereum>()); }
