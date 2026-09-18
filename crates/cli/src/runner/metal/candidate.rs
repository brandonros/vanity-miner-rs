//! Guarded transport for structured cryptographic candidates with variable-length
//! messages. Buffers are allocated per launch; throughput tuning is separate.
use super::transport::{bytes, guarded, load_artifact, record};
use llvm_metal_runtime::{Buffer, Kernel};
use logic::search::{
    candidate_result::{BatchResult, CandidateResult},
    device_record::DeviceRecord,
    hex_pattern::HexPattern,
};
use std::{
    path::Path,
    time::{Duration, Instant},
};
use zeroize::{Zeroize, Zeroizing};

#[repr(C)]
#[derive(Clone, Copy)]
struct Launch {
    start: u64,
    message_len: u64,
    count: u32,
    audit: u32,
}
// SAFETY: repr(C), integer fields with no padding; all bit patterns valid.
unsafe impl DeviceRecord for Launch {}

/// A reviewed kernel and CPU reference sharing the candidate transport ABI.
///
/// # Safety
/// The interface and loaded kernel must use these six disjoint buffers in order:
/// the 24-byte `Launch` header (`u64` start/message_len, `u32` count/audit),
/// one `Request`, one `HexPattern`, `message_len` readable bytes (one allocated
/// byte when empty), one read/write `BatchResult`, and writable audit records.
/// Record layouts/alignment must match the Rust types exactly. The kernel must
/// access only those spans, write at most `count` audit records when audit is
/// nonzero (none otherwise), ignore padded grid lanes >= count, and synchronize
/// shared result writes without data races. It must not mutate input buffers.
/// Every bit pattern admitted by `Request: DeviceRecord` must be handled safely,
/// including malformed requests, and results must use the `CandidateResult` ABI.
/// `INTERFACE` is checked against the artifact before this transport dispatches.
pub unsafe trait Contract {
    type Request: DeviceRecord;
    const INTERFACE: &'static str;
    fn candidate(
        request: &Self::Request,
        pattern: &HexPattern,
        message: &[u8],
        counter: u64,
    ) -> CandidateResult;
}

pub struct CandidateTransport<C: Contract> {
    kernel: Kernel,
    contract: std::marker::PhantomData<C>,
    capacity: u32,
    group: usize,
    audit: bool,
    pub load_time: Duration,
    pub dispatch_time: Duration,
    pub verification_time: Duration,
    pub launches: u64,
}

// Host mirrors can contain private signing keys. Clear on success and every error
// path. Device allocations are released by the runtime after synchronous dispatch.
struct Buffers([Buffer; 6]);
impl Drop for Buffers {
    fn drop(&mut self) {
        for buffer in &mut self.0 {
            buffer.bytes.zeroize();
        }
    }
}
impl<C: Contract> CandidateTransport<C> {
    pub fn load(
        directory: &Path,
        capacity: u32,
        group: usize,
        audit: bool,
    ) -> Result<Self, String> {
        if capacity == 0 || capacity > 1_048_576 || group == 0 || group > 1024 {
            return Err("invalid Metal candidate dispatch size".into());
        }
        let (kernel, load_time) = load_artifact(directory, C::INTERFACE)?;
        Ok(Self {
            kernel,
            contract: std::marker::PhantomData,
            capacity,
            group,
            audit,
            load_time,
            dispatch_time: Duration::ZERO,
            verification_time: Duration::ZERO,
            launches: 0,
        })
    }

    pub fn evaluate(
        &mut self,
        request: &C::Request,
        pattern: &HexPattern,
        message: &[u8],
        start: u64,
        count: u32,
    ) -> Result<BatchResult, String> {
        if count == 0 || count > self.capacity || start.checked_add(u64::from(count) - 1).is_none()
        {
            return Err("invalid Metal candidate range".into());
        }
        let launch = Launch {
            start,
            message_len: message.len() as u64,
            count,
            audit: u32::from(self.audit),
        };
        let sizes = [
            size_of::<Launch>(),
            size_of::<C::Request>(),
            size_of::<HexPattern>(),
            message.len().max(1),
            size_of::<BatchResult>(),
            if self.audit {
                count as usize * size_of::<CandidateResult>()
            } else {
                size_of::<CandidateResult>()
            },
        ];
        let inputs = [bytes(&launch), bytes(request), bytes(pattern), message];
        let mut buffers = Buffers(sizes.map(guarded));
        for (buffer, input) in buffers.0[..4].iter_mut().zip(inputs) {
            buffer.bytes[256..256 + input.len()].copy_from_slice(input);
        }
        buffers.0[4].bytes[256..256 + sizes[4]].copy_from_slice(bytes(&BatchResult::EMPTY));
        let started = Instant::now();
        // SAFETY: reviewed entry, exact artifact ABI, initialized disjoint records,
        // message length and audit capacity match allocation; counter range bounded.
        unsafe {
            self.kernel.run(
                &mut buffers.0,
                (count as usize).div_ceil(self.group) * self.group,
                self.group,
            )?;
        }
        self.dispatch_time += started.elapsed();
        self.launches += 1;
        let checked = Instant::now();
        let used = [
            sizes[0],
            sizes[1],
            sizes[2],
            message.len(),
            sizes[4],
            if self.audit { sizes[5] } else { 0 },
        ];
        for (buffer, used) in buffers.0.iter().zip(used) {
            if buffer.bytes[..256]
                .iter()
                .chain(buffer.bytes[256 + used..].iter())
                .any(|&b| b != 0xa5)
            {
                return Err("Metal kernel changed a guard or unused output byte".into());
            }
        }
        for (buffer, input) in buffers.0[..4].iter().zip(inputs) {
            if &buffer.bytes[256..256 + input.len()] != input {
                return Err("Metal kernel changed an input".into());
            }
        }
        let result: BatchResult = record(&buffers.0[4].bytes[256..256 + sizes[4]]);
        if self.audit {
            let (mut matches, mut errors) = (0, 0);
            for lane in 0..count {
                let expected = Zeroizing::new(C::candidate(
                    request,
                    pattern,
                    message,
                    start + u64::from(lane),
                ));
                let offset = 256 + lane as usize * size_of::<CandidateResult>();
                let actual = Zeroizing::new(record::<CandidateResult>(
                    &buffers.0[5].bytes[offset..offset + size_of::<CandidateResult>()],
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
        if let Some((lane, winner)) = result.winner(count)? {
            let expected = Zeroizing::new(C::candidate(
                request,
                pattern,
                message,
                start + u64::from(lane),
            ));
            if winner.status != expected.status || winner.bytes != expected.bytes {
                return Err("Metal winner differs from CPU reference".into());
            }
        } else if result.lane != u32::MAX
            || bytes(&result.candidate) != bytes(&CandidateResult::MISS)
        {
            return Err("Metal miss changed the unused winner record".into());
        }
        self.verification_time += checked.elapsed();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn launch_layout() {
        assert_eq!(size_of::<super::Launch>(), 24);
        assert_eq!(std::mem::offset_of!(super::Launch, message_len), 8);
        assert_eq!(std::mem::offset_of!(super::Launch, count), 16);
        assert_eq!(std::mem::offset_of!(super::Launch, audit), 20);
    }
}
