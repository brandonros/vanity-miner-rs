use super::Outcome;
use std::error::Error;
pub fn run(args: &super::args::SelfTestArgs) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut results = [0; logic::self_test::SELF_TEST_NUM_CHECKS];
    let cases = args.selected()?;
    let selected: Vec<_> = cases
        .iter()
        .map(|case| logic::self_test::Slot::from_name(case.name).unwrap())
        .collect();
    logic::self_test::run_selected_self_tests(&mut results, &selected);
    super::run("CPU", &cases, |case| {
        let slot = case.slot;
        if results[slot] != 1 {
            return Err("known-answer mismatch".into());
        }
        Ok(Outcome::Passed)
    })
    .map_err(Into::into)
}
