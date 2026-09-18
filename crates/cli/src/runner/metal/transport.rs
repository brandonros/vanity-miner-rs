//! One prepared six-buffer engine for typed requests and patterns.
use super::{
    artifacts::load_artifact,
    buffers::{bytes, guarded, record},
};
use llvm_metal_runtime::{Buffer, LoadTimings, PreparedKernel};
pub use logic::search::candidate_abi::Contract;
use logic::search::{
    candidate_abi::Launch,
    candidate_result::{BatchResult, CandidateResult},
};
use std::{
    path::Path,
    time::{Duration, Instant},
};
use zeroize::{Zeroize, Zeroizing};
#[path = "../../../../kernels/common/candidate_interface.rs"]
mod interface;

struct Buffers([Buffer; 6]);
impl Buffers {
    fn clear(&mut self) {
        for buffer in &mut self.0 {
            buffer.bytes.as_mut_slice().zeroize();
        }
    }
}
impl Drop for Buffers {
    fn drop(&mut self) {
        self.clear();
    }
}

pub struct Transport<C: Contract> {
    kernel: PreparedKernel,
    buffers: Buffers,
    contract: std::marker::PhantomData<C>,
    capacity: u32,
    group: usize,
    audit: bool,
    pub load_time: Duration,
    pub load_stages: LoadTimings,
    pub allocation_time: Duration,
    pub allocation_rounds: u64,
    pub upload_time: Duration,
    pub download_time: Duration,
    pub gpu_time: Duration,
    pub gpu_timed_launches: u64,
    pub dispatch_time: Duration,
    pub verification_time: Duration,
    pub cleanup_time: Duration,
    pub launches: u64,
}
impl<C: Contract> Transport<C> {
    pub fn load(
        directory: &Path,
        capacity: u32,
        group: usize,
        audit: bool,
    ) -> Result<Self, String> {
        if capacity == 0 || capacity > 1_048_576 || group == 0 || group > 1024 {
            return Err("invalid Metal candidate dispatch size".into());
        }
        let (kernel, load_time) = load_artifact(directory, &interface::interface::<C>())?;
        let load_stages = kernel.load_timings();
        let started = Instant::now();
        let mut buffers = Buffers(C::layout().map(|a| guarded(a.bytes)));
        buffers.0[5] = guarded(if audit {
            capacity as usize * size_of::<CandidateResult>()
        } else {
            size_of::<CandidateResult>()
        });
        let kernel = kernel.prepare(&buffers.0)?;
        Ok(Self {
            kernel,
            buffers,
            contract: std::marker::PhantomData,
            capacity,
            group,
            audit,
            load_time,
            load_stages,
            allocation_time: started.elapsed(),
            allocation_rounds: 1,
            upload_time: Duration::ZERO,
            download_time: Duration::ZERO,
            gpu_time: Duration::ZERO,
            gpu_timed_launches: 0,
            dispatch_time: Duration::ZERO,
            verification_time: Duration::ZERO,
            cleanup_time: Duration::ZERO,
            launches: 0,
        })
    }
    /// Identical timing accounting for every candidate mode.
    pub fn print_timings(&self, label: &str) {
        eprintln!(
            "{label}: {} launches; library {:.3} ms; pipeline {:.3} ms; allocation {:.3} ms ({} rounds); upload {:.3} ms; download {:.3} ms; dispatch {:.3} ms; GPU {:.3} ms ({} timed); validation {:.3} ms; cleanup {:.3} ms",
            self.launches,
            self.load_stages.library.as_secs_f64() * 1000.,
            self.load_stages.pipeline.as_secs_f64() * 1000.,
            self.allocation_time.as_secs_f64() * 1000.,
            self.allocation_rounds,
            self.upload_time.as_secs_f64() * 1000.,
            self.download_time.as_secs_f64() * 1000.,
            self.dispatch_time.as_secs_f64() * 1000.,
            self.gpu_time.as_secs_f64() * 1000.,
            self.gpu_timed_launches,
            self.verification_time.as_secs_f64() * 1000.,
            self.cleanup_time.as_secs_f64() * 1000.,
        );
    }
    pub fn evaluate(
        &mut self,
        request: &C::Request,
        pattern: &C::Pattern,
        message: &[u8],
        start: u64,
        count: u32,
    ) -> Result<BatchResult, String> {
        let result = self.evaluate_inner(request, pattern, message, start, count);
        // GPU dispatch is synchronous, including failure. Erase reusable mirrors and
        // retained shared allocations before returning success or any error.
        let started = Instant::now();
        self.buffers.clear();
        self.kernel.clear();
        self.cleanup_time += started.elapsed();
        result
    }
    fn evaluate_inner(
        &mut self,
        request: &C::Request,
        pattern: &C::Pattern,
        message: &[u8],
        start: u64,
        count: u32,
    ) -> Result<BatchResult, String> {
        if count == 0 || count > self.capacity || start.checked_add(u64::from(count) - 1).is_none()
        {
            return Err("invalid Metal candidate range".into());
        }
        C::validate_payload(message)?;
        let launch = Launch {
            start,
            message_len: message.len() as u64,
            count,
            audit: u32::from(self.audit),
        };
        let started = Instant::now();
        if message.len() > self.buffers.0[3].bytes.len() - 512 {
            let length = message
                .len()
                .checked_next_power_of_two()
                .and_then(|n| n.checked_add(512))
                .ok_or("Metal payload is too large")?;
            self.buffers.0[3].bytes.resize(length, 0);
        }
        if self.kernel.reconfigure(&self.buffers.0)? {
            self.allocation_rounds += 1;
        }
        self.allocation_time += started.elapsed();
        for buffer in &mut self.buffers.0 {
            buffer.bytes.fill(0xa5);
        }
        let sizes = [
            size_of::<Launch>(),
            size_of::<C::Request>(),
            size_of::<C::Pattern>(),
            message.len().max(1),
            size_of::<BatchResult>(),
            count as usize * size_of::<CandidateResult>(),
        ];
        let inputs = [bytes(&launch), bytes(request), bytes(pattern), message];
        for (buffer, input) in self.buffers.0[..4].iter_mut().zip(inputs) {
            buffer.bytes[256..256 + input.len()].copy_from_slice(input);
        }
        self.buffers.0[4].bytes[256..256 + sizes[4]].copy_from_slice(bytes(&BatchResult::EMPTY));
        // SAFETY: exact typed ABI, disjoint guarded spans, padded lanes ignored,
        // validated counter range and input sizes, bounded audit capacity.
        let timing = unsafe {
            self.kernel.run(
                &mut self.buffers.0,
                (count as usize).div_ceil(self.group) * self.group,
                self.group,
            )?
        };
        self.dispatch_time += timing.wall;
        self.upload_time += timing.upload;
        self.download_time += timing.download;
        if let Some(gpu) = timing.gpu {
            self.gpu_time += gpu;
            self.gpu_timed_launches += 1;
        }
        self.launches += 1;
        let checked = Instant::now();
        let result = verify::<C>(&self.buffers.0, &launch, request, pattern, message);
        self.verification_time += checked.elapsed();
        result
    }
}

fn verify<C: Contract>(
    buffers: &[Buffer; 6],
    launch: &Launch,
    request: &C::Request,
    pattern: &C::Pattern,
    message: &[u8],
) -> Result<BatchResult, String> {
    let sizes = [
        size_of::<Launch>(),
        size_of::<C::Request>(),
        size_of::<C::Pattern>(),
        message.len().max(1),
        size_of::<BatchResult>(),
        launch.count as usize * size_of::<CandidateResult>(),
    ];
    let inputs = [bytes(launch), bytes(request), bytes(pattern), message];
    let used = [
        sizes[0],
        sizes[1],
        sizes[2],
        message.len(),
        sizes[4],
        if launch.audit != 0 { sizes[5] } else { 0 },
    ];
    for (buffer, used) in buffers.iter().zip(used) {
        if buffer.bytes[..256]
            .iter()
            .chain(buffer.bytes[256 + used..].iter())
            .any(|&b| b != 0xa5)
        {
            return Err("Metal kernel changed a guard or unused output byte".into());
        }
    }
    for (buffer, input) in buffers[..4].iter().zip(inputs) {
        if &buffer.bytes[256..256 + input.len()] != input {
            return Err("Metal kernel changed an input".into());
        }
    }
    let result: BatchResult = record(&buffers[4].bytes[256..256 + sizes[4]]);
    if launch.audit != 0 {
        let (mut matches, mut errors) = (0, 0);
        for lane in 0..launch.count {
            let expected = Zeroizing::new(C::candidate(
                request,
                pattern,
                message,
                launch.start + u64::from(lane),
            ));
            let offset = 256 + lane as usize * size_of::<CandidateResult>();
            let actual = Zeroizing::new(record::<CandidateResult>(
                &buffers[5].bytes[offset..offset + size_of::<CandidateResult>()],
            ));
            if actual.status != expected.status || actual.bytes != expected.bytes {
                return Err(format!("Metal lane {lane} differs from CPU reference"));
            }
            match expected.status {
                CandidateResult::STATUS_MISS => {}
                CandidateResult::STATUS_MATCH => matches += 1,
                _ => errors += 1,
            }
        }
        if (result.matches, result.errors) != (matches, errors) {
            return Err("Metal batch counts differ from CPU reference".into());
        }
    }
    if let Some((lane, winner)) = result.winner(launch.count)? {
        let expected = Zeroizing::new(C::candidate(
            request,
            pattern,
            message,
            launch.start + u64::from(lane),
        ));
        if winner.status != expected.status || winner.bytes != expected.bytes {
            return Err("Metal winner differs from CPU reference".into());
        }
    } else if result.lane != u32::MAX || bytes(&result.candidate) != bytes(&CandidateResult::MISS) {
        return Err("Metal miss changed the unused winner record".into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Toy;
    // SAFETY: used only for CPU-side verification tests; never dispatched.
    unsafe impl Contract for Toy {
        type Request = [u8; 1];
        type Pattern = [u8; 1];
        const ENTRY: &'static str = "test_only";
        fn candidate(r: &[u8; 1], p: &[u8; 1], _: &[u8], _: u64) -> CandidateResult {
            if r == p {
                CandidateResult::matched(r)
            } else {
                CandidateResult::MISS
            }
        }
    }
    fn buffers(launch: &Launch, r: &[u8; 1], p: &[u8; 1], result: &BatchResult) -> [Buffer; 6] {
        let mut buffers = Toy::layout().map(|a| guarded(a.bytes));
        for (buffer, data) in buffers
            .iter_mut()
            .zip([bytes(launch), r, p, &[], bytes(result), &[]])
        {
            buffer.bytes[256..256 + data.len()].copy_from_slice(data);
        }
        buffers
    }
    #[test]
    fn cleanup_preserves_reusable_host_storage() {
        let mut buffers = Buffers(Toy::layout().map(|a| guarded(a.bytes)));
        let shapes = buffers.0.each_ref().map(|b| (b.bytes.len(), b.offset));
        buffers.clear();
        assert_eq!(
            buffers.0.each_ref().map(|b| (b.bytes.len(), b.offset)),
            shapes
        );
        assert!(buffers.0.iter().all(|b| b.bytes.iter().all(|&v| v == 0)));
    }
    #[test]
    fn rejects_guards_inputs_stale_output_and_corrupted_winners_without_audit() {
        let launch = Launch {
            start: 0,
            message_len: 0,
            count: 1,
            audit: 0,
        };
        let winner = BatchResult {
            matches: 1,
            errors: 0,
            lane: 0,
            candidate: CandidateResult::matched(&[7]),
        };
        assert!(
            verify::<Toy>(
                &buffers(&launch, &[7], &[7], &winner),
                &launch,
                &[7],
                &[7],
                &[]
            )
            .is_ok()
        );
        for index in 0..6 {
            let mut b = buffers(&launch, &[7], &[7], &winner);
            b[index].bytes[0] ^= 1;
            assert!(
                verify::<Toy>(&b, &launch, &[7], &[7], &[])
                    .err()
                    .unwrap()
                    .contains("guard")
            );
        }
        for index in 0..3 {
            let mut b = buffers(&launch, &[7], &[7], &winner);
            b[index].bytes[256] ^= 1;
            assert!(
                verify::<Toy>(&b, &launch, &[7], &[7], &[])
                    .err()
                    .unwrap()
                    .contains("input")
            );
        }
        let mut corrupt = winner;
        corrupt.candidate.bytes[0] ^= 1;
        assert!(
            verify::<Toy>(
                &buffers(&launch, &[7], &[7], &corrupt),
                &launch,
                &[7],
                &[7],
                &[]
            )
            .is_err()
        );
        let stale = BatchResult {
            matches: 0,
            ..winner
        };
        assert!(
            verify::<Toy>(
                &buffers(&launch, &[7], &[8], &stale),
                &launch,
                &[7],
                &[8],
                &[]
            )
            .is_err()
        );
        let b = buffers(&launch, &[7], &[8], &BatchResult::EMPTY);
        assert!(verify::<Toy>(&b, &launch, &[7], &[8], &[]).is_ok());
    }
}
