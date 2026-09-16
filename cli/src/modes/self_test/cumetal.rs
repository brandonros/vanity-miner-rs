use crate::runner::cumetal::{CumetalRunner, Error, driver::Driver};
use std::rc::Rc;

pub(crate) fn run(
    runner: &CumetalRunner,
    driver: &Rc<Driver>,
    args: &super::args::SelfTestArgs,
) -> Result<(), Error> {
    use crate::modes::self_test;
    let cases = args.selected()?;
    let mut cache = self_test::DeviceResults::default();
    self_test::run("CuMetal", &cases, |case| {
        cache.check(case, || {
            let operation = (|| -> Result<Vec<u32>, Error> {
                let module = runner.module(driver, case.kernel)?;
                let result =
                    driver.buffer(&vec![0xa5; logic::self_test::SELF_TEST_NUM_CHECKS * 4])?;
                module.launch(&mut [result.pointer()], 1, 1)?;
                let bytes = result.read()?;
                Ok(bytes
                    .chunks_exact(4)
                    .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                    .collect())
            })();
            operation.map_err(|e| e.to_string())
        })
    })
    .map_err(Into::into)
}
