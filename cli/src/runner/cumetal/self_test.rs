use super::{CumetalRunner, Error, driver::Driver};
use std::rc::Rc;

impl CumetalRunner {
    pub(super) fn self_tests(&self, driver: &Rc<Driver>) -> Result<(), Error> {
        use vanity_miner::self_test_suite::{self, Outcome};
        let mut cache = self_test_suite::DeviceResults::default();
        self_test_suite::run("CuMetal", |case| {
            if !self.options.self_test_slot.is_empty()
                && !self.options.self_test_slot.contains(&(case.slot as u32))
            {
                return Ok(Outcome::Skipped("not selected"));
            }
            cache.check(case, || {
                let operation = (|| -> Result<Vec<u32>, Error> {
                    let module = self.module(driver, case.kernel)?;
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
}
