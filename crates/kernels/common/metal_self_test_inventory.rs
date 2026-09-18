//! Machine-readable inventory generated from the original registry, never parsed from Rust source.
fn main() {
    for case in logic::self_test::metadata::CASES
        .iter()
        .filter(|case| case.enabled)
    {
        println!(
            "{}\t{}\t{}\t{}",
            case.slot, case.name, case.kernel, case.metal_entry
        );
    }
}
