//! CuMetal executes the same resumable per-thread RSA miner as CUDA.
use crate::runner::cumetal::{
    CumetalRunner, Error,
    driver::{Buffer, Driver, Module, record_bytes},
};
use crate::runner::{progress::GlobalStats, session::run_device_session};
use logic::{
    modes::rsa_modulus::{self as mining, Counts, Pair, SearchConfig, Task},
    search::hex_pattern::HexPattern,
};
use std::{rc::Rc, sync::Arc};
use zeroize::Zeroizing;

struct RsaPipeline {
    mine: Module,
    config: Buffer,
    pattern: Buffer,
    tasks: Buffer,
    pairs: Buffer,
    counts: Buffer,
    capacity: u32,
    steps: u32,
    threads: u32,
    verify: bool,
}
impl RsaPipeline {
    fn new(
        runner: &CumetalRunner,
        driver: &Rc<Driver>,
        config: &SearchConfig,
        pattern: &HexPattern,
        capacity: u32,
        steps: u32,
    ) -> Result<Self, Error> {
        mining::launch_work(capacity, steps)?;
        Ok(Self {
            mine: runner.module(driver, mining::ENTRY)?,
            config: driver.buffer(record_bytes(&[*config]))?,
            pattern: driver.buffer(record_bytes(&[*pattern]))?,
            tasks: driver.buffer(record_bytes(&vec![Task::EMPTY; capacity as usize]))?,
            pairs: driver.buffer(record_bytes(&vec![Pair::EMPTY; capacity as usize]))?,
            counts: driver.buffer(record_bytes(&[Counts::default()]))?,
            capacity,
            steps,
            threads: runner.options.threads_per_block,
            verify: runner.options.verify,
        })
    }
    fn cycle(
        &mut self,
        config: &SearchConfig,
        pattern: &HexPattern,
        start: u64,
    ) -> Result<(Counts, Zeroizing<Vec<Pair>>), Error> {
        let work = mining::launch_work(self.capacity, self.steps)?;
        start
            .checked_add(u64::from(work) - 1)
            .ok_or("RSA task counter exhausted")?;
        let expected = if self.verify {
            Some(self.reference_cycle(config, pattern, start)?)
        } else {
            None
        };
        self.counts.write(record_bytes(&[Counts::default()]))?;
        self.mine.launch(
            &mut [
                self.config.pointer(),
                self.pattern.pointer(),
                self.tasks.pointer(),
                self.pairs.pointer(),
                start,
                u64::from(self.capacity),
                u64::from(self.steps),
                self.counts.pointer(),
            ],
            self.capacity.div_ceil(self.threads),
            self.threads,
        )?;
        let counts = self.counts.read_records::<Counts>(1)?[0];
        counts.validate(self.capacity, self.steps)?;
        let mut pairs = self.pairs.read_records::<Pair>(counts.matches as usize)?;
        if let Some((expected_counts, mut expected_pairs, expected_tasks)) = expected {
            pairs.sort_by_key(|pair| pair.id);
            expected_pairs.sort_by_key(|pair| pair.id);
            let tasks = self.tasks.read_records::<Task>(self.capacity as usize)?;
            if counts != expected_counts || *pairs != *expected_pairs || *tasks != *expected_tasks {
                return Err(
                    "CuMetal RSA miner differs from CPU reference (counts, pairs or resumed state)"
                        .into(),
                );
            }
        } else {
            // Keep task state resident; inspecting the guard needs only 32 bytes.
            self.tasks.check_guards()?;
        }
        self.pairs
            .write(record_bytes(&vec![Pair::EMPTY; self.capacity as usize]))?;
        if Zeroizing::new(self.config.read()?).as_slice() != record_bytes(&[*config])
            || self.pattern.read()? != record_bytes(&[*pattern])
        {
            return Err("CuMetal RSA miner changed an input".into());
        }
        Ok((counts, pairs))
    }
    fn reference_cycle(
        &self,
        config: &SearchConfig,
        pattern: &HexPattern,
        start: u64,
    ) -> Result<(Counts, Zeroizing<Vec<Pair>>, Zeroizing<Vec<Task>>), Error> {
        let mut tasks = self.tasks.read_records::<Task>(self.capacity as usize)?;
        let mut counts = Counts::default();
        let mut pairs = Zeroizing::new(Vec::new());
        for (lane, task) in tasks.iter_mut().enumerate() {
            let (local, pair) = mining::mine(
                config,
                pattern,
                task,
                start + lane as u64,
                self.capacity,
                self.steps,
            );
            macro_rules! sum {
                ($($field:ident),+) => { $(counts.$field += local.$field;)+ };
            }
            sum!(
                p_tested, p_accepted, ranges, q_tested, matches, errors, active
            );
            if let Some(pair) = pair {
                pairs.push(pair);
            }
        }
        Ok((counts, pairs, tasks))
    }
}

pub fn run(
    runner: &CumetalRunner,
    args: &super::args::RsaModulusArgs,
    driver: &Rc<Driver>,
    stats: Arc<GlobalStats>,
) -> Result<(), Error> {
    let config = args.config(1)?;
    let stages = super::pipeline::StageStats::attach(&stats)?;
    let mut engine = None;
    run_device_session(
        stats,
        "factor candidates (p + q)",
        runner.options.batches,
        runner.options.blocks * runner.options.threads_per_block,
        |control| {
            super::pipeline::run(
                &config,
                &control,
                &stages,
                args.steps_per_launch,
                |r, p, start, capacity| {
                    if engine.is_none() {
                        engine = Some(
                            RsaPipeline::new(runner, driver, r, p, capacity, args.steps_per_launch)
                                .map_err(|e| e.to_string())?,
                        );
                    }
                    engine
                        .as_mut()
                        .unwrap()
                        .cycle(r, p, start)
                        .map_err(|e| e.to_string())
                },
            )
        },
    )
}
