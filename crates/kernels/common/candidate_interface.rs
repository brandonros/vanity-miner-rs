//! Native inspection form of the shared typed contract. Device-embedded descriptor
//! extraction remains llvm-metal#1 / vanity-miner-rs#42.
pub fn interface<C: logic::search::candidate_abi::Contract>() -> String {
    let arguments: Vec<_> = C::layout()
        .iter()
        .map(|a| {
            format!(
                r#"{{"name":"{}","kind":"buffer","access":"{}","bytes":{},"alignment":{}}}"#,
                a.name, a.access, a.bytes, a.alignment
            )
        })
        .collect();
    format!(
        r#"{{"schema":1,"entry":"{}","calling_convention":"C","invocations":null,"dispatch":"grid1d","aliasing":"all buffers disjoint","arguments":[{}]}}"#,
        C::ENTRY,
        arguments.join(",")
    )
}
