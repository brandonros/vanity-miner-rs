use super as self_test;
use std::error::Error;
pub fn run(args: &super::args::SelfTestArgs) -> Result<(), Box<dyn Error + Send + Sync>> {
    self_test::run("CPU", &args.selected()?, |mode| {
        let mut results = vec![self_test::SENTINEL; mode.checks.len()];
        (mode.run)(&mut results);
        Ok(results)
    })
    .map_err(Into::into)
}
