//! CuMetal executes the same persistent RSA stages and records as CUDA.
use crate::runner::cumetal::{
    CumetalRunner, Error,
    driver::{Buffer, Driver, Module, record_bytes},
};
use crate::runner::{progress::GlobalStats, session::run_device_session};
use logic::{
    modes::rsa_modulus::{Counts, Pair, SearchConfig, Task},
    search::hex_pattern::HexPattern,
};
use std::{rc::Rc, sync::Arc};
use zeroize::Zeroizing;

struct RsaPipeline {
    generate: Module,
    ranges: Module,
    search: Module,
    advance: Module,
    config: Buffer,
    pattern: Buffer,
    tasks: Buffer,
    active: Buffer,
    winners: Buffer,
    pairs: Buffer,
    counts: Buffer,
    capacity: u32,
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
    ) -> Result<Self, Error> {
        Ok(Self {
            generate: runner.module(driver, "kernel_rsa_generate")?,
            ranges: runner.module(driver, "kernel_rsa_ranges")?,
            search: runner.module(driver, "kernel_rsa_search")?,
            advance: runner.module(driver, "kernel_rsa_advance")?,
            config: driver.buffer(record_bytes(&[*config]))?,
            pattern: driver.buffer(record_bytes(&[*pattern]))?,
            tasks: driver.buffer(record_bytes(&vec![Task::EMPTY; capacity as usize]))?,
            pairs: driver.buffer(record_bytes(&vec![Pair::EMPTY; capacity as usize]))?,
            counts: driver.buffer(record_bytes(&[Counts::default()]))?,
            active: driver.buffer(&vec![0; capacity as usize * 4])?,
            winners: driver.buffer(&vec![0; capacity as usize * 4])?,
            capacity,
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
        let capacity = u64::from(self.capacity);
        start
            .checked_add(capacity - 1)
            .ok_or("RSA task counter exhausted")?;
        let blocks = self.capacity.div_ceil(self.threads);
        self.counts.write(record_bytes(&[Counts::default()]))?;
        self.generate.launch(
            &mut [
                self.config.pointer(),
                self.tasks.pointer(),
                start,
                capacity,
                self.counts.pointer(),
            ],
            blocks,
            self.threads,
        )?;
        self.ranges.launch(
            &mut [
                self.config.pointer(),
                self.tasks.pointer(),
                self.active.pointer(),
                self.winners.pointer(),
                capacity,
                self.counts.pointer(),
            ],
            blocks,
            self.threads,
        )?;
        let expected = if self.verify {
            Some(self.reference_candidates(config, pattern)?)
        } else {
            None
        };
        self.search.launch(
            &mut [
                self.config.pointer(),
                self.pattern.pointer(),
                self.tasks.pointer(),
                self.active.pointer(),
                self.winners.pointer(),
                self.pairs.pointer(),
                capacity,
                self.counts.pointer(),
            ],
            blocks,
            self.threads,
        )?;
        self.advance.launch(
            &mut [
                self.tasks.pointer(),
                self.active.pointer(),
                self.winners.pointer(),
                capacity,
                self.counts.pointer(),
            ],
            blocks,
            self.threads,
        )?;
        let counts = self.counts.read_records::<Counts>(1)?[0];
        if counts.errors != 0 || counts.matches > self.capacity {
            return Err("RSA pipeline returned invalid counters".into());
        }
        let pairs = self.pairs.read_records::<Pair>(counts.matches as usize)?;
        if let Some((q_tested, matches)) = expected {
            if q_tested != counts.q_tested
                || matches.len() != pairs.len()
                || pairs.iter().any(|pair| {
                    !matches
                        .iter()
                        .any(|(id, qs)| *id == pair.id && qs.contains(&pair.q))
                })
            {
                return Err("CuMetal RSA search differs from CPU reference".into());
            }
        }
        self.pairs
            .write(record_bytes(&vec![Pair::EMPTY; self.capacity as usize]))?;
        // Check every guard, including read-only inputs and compact work queues.
        for buffer in [
            &self.config,
            &self.pattern,
            &self.tasks,
            &self.active,
            &self.winners,
        ] {
            let _guard = Zeroizing::new(buffer.read()?);
        }
        Ok((counts, pairs))
    }
    fn reference_candidates(
        &self,
        config: &SearchConfig,
        pattern: &HexPattern,
    ) -> Result<(u32, Zeroizing<Vec<(u64, Vec<[u8; 128]>)>>), Error> {
        use logic::modes::rsa_modulus as pipeline;
        let counts = self.counts.read_records::<Counts>(1)?[0];
        if counts.errors != 0 || counts.active > self.capacity {
            return Err("invalid RSA active count".into());
        }
        let tasks = self.tasks.read_records::<Task>(self.capacity as usize)?;
        let raw = Zeroizing::new(self.active.read()?);
        let active: Vec<_> = raw
            .chunks_exact(4)
            .take(counts.active as usize)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
            .collect();
        let mut matches: Zeroizing<Vec<(u64, Vec<[u8; 128]>)>> = Zeroizing::new(Vec::new());
        let mut tested = 0;
        if active.is_empty() {
            return Ok((tested, matches));
        }
        for lane in 0..self.capacity {
            let task = tasks
                .get(active[lane as usize % active.len()])
                .ok_or("invalid RSA task slot")?;
            if let Some(q) = pipeline::q_at(config, task, lane / counts.active) {
                tested += 1;
                if pipeline::eligible_pair(&task.p, &q, pattern) {
                    if let Some((_, qs)) = matches.iter_mut().find(|(id, _)| *id == task.id) {
                        qs.push(q);
                    } else {
                        matches.push((task.id, vec![q]));
                    }
                }
            }
        }
        Ok((tested, matches))
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
            super::pipeline::run(&config, &control, &stages, |r, p, start, capacity| {
                if engine.is_none() {
                    engine = Some(
                        RsaPipeline::new(runner, driver, r, p, capacity)
                            .map_err(|e| e.to_string())?,
                    );
                }
                engine
                    .as_mut()
                    .unwrap()
                    .cycle(r, p, start)
                    .map_err(|e| e.to_string())
            })
        },
    )
}
