//! The group `llvm-metalc build --cases` reads, generated from the original
//! registry, never parsed from Rust source.
fn main() {
    let cases: Vec<_> = logic::self_test::metadata::CASES
        .iter()
        .filter(|case| case.enabled)
        .collect();
    let kernel = cases.first().expect("empty self-test inventory").kernel;
    assert!(
        cases.iter().all(|case| case.kernel == kernel),
        "self-test inventory contains a different group"
    );
    // Names and entries are Rust identifiers and paths, which need no JSON escapes.
    let listed: Vec<_> = cases
        .iter()
        .map(|case| {
            format!(
                r#"{{"slot":{},"name":"{}","entry":"{}"}}"#,
                case.slot, case.name, case.metal_entry
            )
        })
        .collect();
    println!(r#"{{"kernel":"{kernel}","cases":[{}]}}"#, listed.join(","));
}
