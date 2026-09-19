//! Connect the generated entry bindings to the search engine's semantic roles.
macro_rules! candidate_bindings {
    () => {
        const ENTRY: &'static str = abi::ENTRY;
        fn descriptor() -> &'static [u8] {
            &abi::DESCRIPTOR
        }
        fn slots() -> logic::search::candidate_abi::Arguments<usize> {
            let a = abi::Arguments::from_fn(|index, _| index);
            logic::search::candidate_abi::Arguments {
                launch: a.launch,
                request: a.request,
                pattern: a.pattern,
                message: a.message,
                output: a.output,
                records: a.records,
            }
        }
        fn pack<T>(a: logic::search::candidate_abi::Arguments<T>) -> [T; 6] {
            abi::Arguments {
                launch: a.launch,
                request: a.request,
                pattern: a.pattern,
                message: a.message,
                output: a.output,
                records: a.records,
            }
            .into_array()
        }
    };
}
