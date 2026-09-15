//! Transport for the legacy address/nonce buffer ABI.
use super::{CumetalRunner, Error, driver::Driver};
use crate::runner::progress::GlobalStats;
use std::{rc::Rc, sync::Arc};

pub(crate) struct Expected {
    pub(crate) matched: bool,
    pub(crate) payloads: Vec<Vec<u8>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
// A feature subset can use only one of the two supported buffer layouts.
#[allow(dead_code)]
pub(crate) enum ParameterLayout {
    Patterns,
    Shallenge,
}

/// A single prepared launch description. Modes own input encoding, reference
/// evaluation and printing; transport owns allocation, guards and launch order.
pub(crate) struct AddressBatch {
    pub entry: &'static str,
    pub payload_sizes: &'static [usize],
    pub first: Vec<u8>,
    pub second: Vec<u8>,
    pub layout: ParameterLayout,
    pub reference: fn(&[u8], &[u8], u64, usize) -> Result<Expected, Error>,
    pub print: fn(&[Vec<u8>]) -> Result<(), Error>,
}

impl CumetalRunner {
    pub(crate) fn address_search(
        &self,
        mut search: AddressBatch,
        driver: &Rc<Driver>,
        stats: Arc<GlobalStats>,
    ) -> Result<(), Error> {
        let module = self.module(&driver, search.entry)?;
        let candidates = self.options.blocks * self.options.threads_per_block;
        let mut batch = 0u64;
        loop {
            let seed = self
                .options
                .seed
                .map(|s| s.wrapping_add(batch))
                .unwrap_or_else(rand::random);
            let (first, second) = (&search.first, &search.second);
            let first_buffer = driver.buffer(first)?;
            let second_buffer = driver.buffer(second)?;
            let count = driver.buffer(&0u32.to_le_bytes())?;
            let payloads: Vec<_> = search
                .payload_sizes
                .iter()
                .map(|n| driver.buffer(&vec![0xa5; *n]))
                .collect::<Result<_, _>>()?;
            let thread = driver.buffer(&[0xa5; 4])?;
            #[allow(unused_mut)]
            let mut args = vec![
                first_buffer.pointer(),
                first.len() as u64,
                second_buffer.pointer(),
                second.len() as u64,
                seed,
            ];
            if search.layout == ParameterLayout::Shallenge {
                args = vec![
                    first_buffer.pointer(),
                    first.len() as u64,
                    second_buffer.pointer(),
                    seed,
                ];
            }
            args.push(count.pointer());
            args.extend(payloads.iter().map(|b| b.pointer()));
            args.push(thread.pointer());
            module.launch(
                &mut args,
                self.options.blocks,
                self.options.threads_per_block,
            )?;
            if first_buffer.read()? != *first || second_buffer.read()? != *second {
                return Err("kernel changed an input buffer".into());
            }
            let matches = u32::from_le_bytes(count.read()?.try_into().unwrap());
            if matches > candidates {
                return Err("match count exceeds candidate count".into());
            }
            let output: Vec<_> = payloads
                .iter()
                .map(|b| b.read())
                .collect::<Result<_, _>>()?;
            let thread_bytes = thread.read()?;
            if self.options.verify {
                let mut expected_count = 0;
                for i in 0..candidates {
                    if (search.reference)(first, second, seed, i as usize)?.matched {
                        expected_count += 1;
                    }
                }
                if matches != expected_count {
                    return Err(format!("match count: GPU {matches}, CPU {expected_count}").into());
                }
            }
            if matches == 0 {
                if output
                    .iter()
                    .flatten()
                    .chain(thread_bytes.iter())
                    .any(|b| *b != 0xa5)
                {
                    return Err("no-match launch modified a result payload".into());
                }
            } else {
                let selected = u32::from_le_bytes(thread_bytes.try_into().unwrap());
                if selected >= candidates {
                    return Err("returned thread index outside launch".into());
                }
                // Always validate the returned key/nonce, even without a full CPU sweep.
                let reference = (search.reference)(first, second, seed, selected as usize)?;
                if !reference.matched {
                    return Err(format!("GPU reported a nonmatching CPU candidate for thread {selected}, seed {seed}").into());
                }
                if output.len() != reference.payloads.len() {
                    return Err("GPU output payload count differs from CPU reference".into());
                }
                let differences: Vec<_> = output
                    .iter()
                    .zip(&reference.payloads)
                    .enumerate()
                    .filter(|(_, (actual, expected))| actual != expected)
                    .map(|(index, (actual, expected))| {
                        let offset = actual.iter().zip(expected).position(|(a, b)| a != b);
                        format!(
                            "payload {index}: first differing byte {offset:?}, lengths {}/{}",
                            actual.len(),
                            expected.len()
                        )
                    })
                    .collect();
                if !differences.is_empty() {
                    return Err(format!(
                        "GPU result differs from CPU for thread {selected}, seed {seed}: {}",
                        differences.join("; ")
                    )
                    .into());
                }
                println!("MATCH thread={selected} seed={seed}");
                (search.print)(&output)?;
                if search.layout == ParameterLayout::Shallenge {
                    search.second = output[0].clone();
                }
            }
            stats.add_launch(candidates as usize);
            stats.add_matches(matches as usize);
            println!(
                "CUMETAL_BATCH batch={batch} seed={seed} candidates={candidates} matches={matches} verified={} guards=intact",
                self.options.verify
            );
            batch += 1;
            if self.options.batches.is_some_and(|limit| batch >= limit) {
                break;
            }
        }
        Ok(())
    }
}
