#[path = "../../common/candidate_interface.rs"]
mod interface;
fn main() { println!("{}", interface::interface::<kernel_bitcoin_metal::Bitcoin>()); }
