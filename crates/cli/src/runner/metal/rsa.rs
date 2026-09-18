//! Resumable RSA transport. Task records survive launches through checked
//! readback/upload; the current runtime transfers every buffer on each dispatch.
use super::{
    rsa_contract::{INTERFACE, Launch},
    transport::{bytes, guarded, load_artifact, record},
};
use llvm_metal_runtime::{Buffer, PreparedKernel};
use logic::{
    modes::rsa_modulus::{self as mining, Counts, Pair, SearchConfig, Task},
    search::hex_pattern::HexPattern,
};
use std::{
    path::Path,
    time::{Duration, Instant},
};
use zeroize::{Zeroize, Zeroizing};

pub struct RsaTransport {
    kernel: PreparedKernel,
    buffers: [Buffer; 6],
    capacity: u32,
    steps: u32,
    group: usize,
    audit: bool,
    pub load_time: Duration,
    pub dispatch_time: Duration,
    pub verification_time: Duration,
    pub launches: u64,
}
impl RsaTransport {
    pub fn load(
        directory: &Path,
        config: &SearchConfig,
        pattern: &HexPattern,
        capacity: u32,
        steps: u32,
        group: usize,
        audit: bool,
    ) -> Result<Self, String> {
        mining::launch_work(capacity, steps)?;
        if group == 0 || group > 1024 {
            return Err("invalid RSA Metal threadgroup size".into());
        }
        let (kernel, load_time) = load_artifact(directory, INTERFACE)?;
        let sizes = [
            size_of::<Launch>(),
            size_of::<SearchConfig>(),
            size_of::<HexPattern>(),
            capacity as usize * size_of::<Task>(),
            capacity as usize * size_of::<Pair>(),
            size_of::<Counts>(),
        ];
        let mut buffers = sizes.map(guarded);
        buffers[1].bytes[256..256 + sizes[1]].copy_from_slice(bytes(config));
        buffers[2].bytes[256..256 + sizes[2]].copy_from_slice(bytes(pattern));
        for chunk in buffers[3].bytes[256..256 + sizes[3]]
            .as_chunks_mut::<{ size_of::<Task>() }>()
            .0
        {
            chunk.copy_from_slice(bytes(&Task::EMPTY));
        }
        let kernel = kernel.prepare(&buffers)?;
        Ok(Self {
            kernel,
            buffers,
            capacity,
            steps,
            group,
            audit,
            load_time,
            dispatch_time: Duration::ZERO,
            verification_time: Duration::ZERO,
            launches: 0,
        })
    }
    pub fn cycle(
        &mut self,
        config: &SearchConfig,
        pattern: &HexPattern,
        start: u64,
    ) -> Result<(Counts, Zeroizing<Vec<Pair>>), String> {
        let work = mining::launch_work(self.capacity, self.steps)?;
        start
            .checked_add(u64::from(work) - 1)
            .ok_or("RSA task counter exhausted")?;
        if self.body(1) != bytes(config) || self.body(2) != bytes(pattern) {
            return Err("RSA search configuration changed between launches".into());
        }
        let audit_start = Instant::now();
        let expected = if self.audit {
            let mut tasks = Zeroizing::new(
                self.body(3)
                    .as_chunks::<{ size_of::<Task>() }>()
                    .0
                    .iter()
                    .map(|raw| record::<Task>(raw))
                    .collect::<Vec<_>>(),
            );
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
                macro_rules! sum {($($field:ident),+)=>{$(counts.$field+=local.$field;)+};}
                sum!(
                    p_tested, p_accepted, ranges, q_tested, matches, errors, active
                );
                if let Some(pair) = pair {
                    pairs.push(pair);
                }
            }
            Some((counts, pairs, tasks))
        } else {
            None
        };
        self.verification_time += audit_start.elapsed();
        let header = Launch {
            start,
            capacity: self.capacity,
            steps: self.steps,
        };
        self.body_mut(0).copy_from_slice(bytes(&header));
        self.body_mut(4).fill(0);
        self.body_mut(5).copy_from_slice(bytes(&Counts::default()));
        // SAFETY: exact ABI, bounded capacity/steps and counter reservation,
        // disjoint guarded buffers, initialized persistent tasks and atomic counts.
        let timing = unsafe {
            self.kernel.run(
                &mut self.buffers,
                (self.capacity as usize).div_ceil(self.group) * self.group,
                self.group,
            )?
        };
        self.dispatch_time += timing.wall;
        self.launches += 1;
        let check = Instant::now();
        for buffer in &self.buffers {
            if buffer.bytes[..256]
                .iter()
                .chain(buffer.bytes[buffer.bytes.len() - 256..].iter())
                .any(|&b| b != 0xa5)
            {
                return Err("RSA Metal kernel changed a guard".into());
            }
        }
        if self.body(0) != bytes(&header)
            || self.body(1) != bytes(config)
            || self.body(2) != bytes(pattern)
        {
            return Err("RSA Metal kernel changed an input".into());
        }
        let counts: Counts = record(self.body(5));
        counts.validate(self.capacity, self.steps)?;
        let used = counts.matches as usize * size_of::<Pair>();
        if self.body(4)[used..].iter().any(|&b| b != 0) {
            return Err("RSA Metal kernel wrote an unused pair slot".into());
        }
        let mut pairs = Zeroizing::new(
            self.body(4)[..used]
                .as_chunks::<{ size_of::<Pair>() }>()
                .0
                .iter()
                .map(|raw| record::<Pair>(raw))
                .collect::<Vec<_>>(),
        );
        pairs.sort_by_key(|p| p.id);
        if let Some((expected_counts, mut expected_pairs, expected_tasks)) = expected {
            expected_pairs.sort_by_key(|p| p.id);
            let tasks_match = self
                .body(3)
                .as_chunks::<{ size_of::<Task>() }>()
                .0
                .iter()
                .zip(expected_tasks.iter())
                .all(|(actual, expected)| actual == bytes(expected));
            if counts != expected_counts || *pairs != *expected_pairs || !tasks_match {
                return Err(
                    "RSA Metal miner differs from CPU reference (counts, pairs or resumed state)"
                        .into(),
                );
            }
        }
        self.verification_time += check.elapsed();
        Ok((counts, pairs))
    }
    fn body(&self, index: usize) -> &[u8] {
        let bytes = &self.buffers[index].bytes;
        &bytes[256..bytes.len() - 256]
    }
    fn body_mut(&mut self, index: usize) -> &mut [u8] {
        let bytes = &mut self.buffers[index].bytes;
        let end = bytes.len() - 256;
        &mut bytes[256..end]
    }
}
impl Drop for RsaTransport {
    fn drop(&mut self) {
        for buffer in &mut self.buffers {
            buffer.bytes.zeroize();
        }
    }
}
