use super::{self as self_test, Outcome};
use std::error::Error;
pub fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut results = [0; logic::self_test::SELF_TEST_NUM_CHECKS];
    logic::self_test::run_self_test(&mut results);
    self_test::run("CPU", |case| {
        let slot = case.slot;
        if results[slot] != 1 {
            return Err("known-answer mismatch".into());
        }
        Ok(Outcome::Passed)
    })
    .map_err(Into::into)
}
