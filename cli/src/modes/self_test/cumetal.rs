use crate::runner::cumetal::{CumetalRunner, Error, driver::Driver};
use std::rc::Rc;

pub(crate) fn run(runner: &CumetalRunner, driver: &Rc<Driver>) -> Result<(), Error> {
    use crate::modes::self_test::{self, Outcome};
    let mut cache = self_test::DeviceResults::default();
    self_test::run("CuMetal", |case| {
        if !runner.options.self_test_slot.is_empty()
            && !runner.options.self_test_slot.contains(&(case.slot as u32))
        {
            return Ok(Outcome::Skipped("not selected"));
        }
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
