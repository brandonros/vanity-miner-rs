use super::contract::{self, Launch};
use llvm_metal_runtime::{Buffer, Kernel, LoadTimings, PreparedKernel};
use logic::search::{
    candidate_result::{BatchResult, CandidateResult},
    device_record::DeviceRecord,
    xoroshiro::BatchSeed,
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

fn bytes<T: DeviceRecord>(value: &T) -> &[u8] {
    // SAFETY: DeviceRecord guarantees initialized padding-free storage.
    unsafe {
        std::slice::from_raw_parts(std::ptr::from_ref(value).cast(), std::mem::size_of::<T>())
    }
}
fn record<T: DeviceRecord>(bytes: &[u8]) -> T {
    assert_eq!(bytes.len(), std::mem::size_of::<T>());
    // SAFETY: exact-size bytes, any bit pattern valid; alignment is not assumed.
    unsafe { bytes.as_ptr().cast::<T>().read_unaligned() }
}
fn guarded(size: usize) -> Buffer {
    Buffer {
        bytes: vec![0xa5; size + 512],
        offset: 256,
    }
}

pub trait Contract {
    type Launch: DeviceRecord;
    const INTERFACE: &'static str;
    fn candidate(launch: &Self::Launch, lane: u32) -> CandidateResult;
}

pub struct Shallenge;
impl Contract for Shallenge {
    type Launch = Launch;
    const INTERFACE: &'static str = contract::INTERFACE;
    fn candidate(launch: &Launch, lane: u32) -> CandidateResult {
        contract::candidate(launch, lane)
    }
}
pub type ShallengeTransport = Transport<Shallenge>;

#[cfg(feature = "ethereum")]
pub struct Ethereum;
#[cfg(feature = "ethereum")]
impl Contract for Ethereum {
    type Launch = super::ethereum_contract::Launch;
    const INTERFACE: &'static str = super::ethereum_contract::INTERFACE;
    fn candidate(launch: &Self::Launch, lane: u32) -> CandidateResult {
        super::ethereum_contract::candidate(launch, lane)
    }
}
#[cfg(feature = "ethereum")]
pub type EthereumTransport = Transport<Ethereum>;

#[cfg(feature = "bitcoin")]
pub struct Bitcoin;
#[cfg(feature = "bitcoin")]
impl Contract for Bitcoin {
    type Launch = super::bitcoin_contract::Launch;
    const INTERFACE: &'static str = super::bitcoin_contract::INTERFACE;
    fn candidate(launch: &Self::Launch, lane: u32) -> CandidateResult {
        super::bitcoin_contract::candidate(launch, lane)
    }
}
#[cfg(feature = "bitcoin")]
pub type BitcoinTransport = Transport<Bitcoin>;

pub struct Transport<C: Contract> {
    contract: std::marker::PhantomData<C>,
    kernel: PreparedKernel,
    buffers: [Buffer; 3],
    audit: bool,
    capacity: u32,
    group: usize,
    pub load_time: Duration,
    pub load_stages: LoadTimings,
    pub allocation_time: Duration,
    pub upload_time: Duration,
    pub download_time: Duration,
    pub gpu_time: Duration,
    pub gpu_timed_launches: u64,
    pub dispatch_time: Duration,
    pub verification_time: Duration,
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
            return Err("invalid batch capacity".into());
        }
        let read = |name| fs::read(directory.join(name)).map_err(|e| format!("{name}: {e}"));
        let manifest: serde_json::Value =
            serde_json::from_slice(&read("kernel.build.json")?).map_err(|e| e.to_string())?;
        if manifest["schema"] != 1 {
            return Err("unsupported Metal artifact schema".into());
        }
        for name in ["kernel.metallib", "kernel.bindings.json"] {
            if manifest["artifacts"][name].as_str()
                != Some(hex::encode(Sha256::digest(read(name)?)).as_str())
            {
                return Err(format!("Metal artifact hash mismatch: {name}"));
            }
        }
        let bindings: llvm_metal_abi::MetalBindings =
            serde_json::from_slice(&read("kernel.bindings.json")?).map_err(|e| e.to_string())?;
        let interface: llvm_metal_abi::KernelInterface =
            serde_json::from_str(C::INTERFACE).map_err(|e| e.to_string())?;
        if serde_json::to_value(&bindings).unwrap()
            != serde_json::to_value(interface.validate()?).unwrap()
        {
            return Err("Metal kernel bindings do not match the application ABI".into());
        }
        let start = Instant::now();
        let kernel = Kernel::load(&directory.join("kernel.metallib"), &bindings)?;
        let load_time = start.elapsed();
        eprintln!(
            "Metal device: {}; library/pipeline load {:.3} ms",
            kernel.device_name(),
            load_time.as_secs_f64() * 1000.
        );
        let load_stages = kernel.load_timings();
        let allocation_start = Instant::now();
        let buffers = [
            guarded(std::mem::size_of::<C::Launch>()),
            guarded(272),
            guarded(if audit { capacity as usize * 260 } else { 260 }),
        ];
        let kernel = kernel.prepare(&buffers)?;
        let allocation_time = allocation_start.elapsed();
        Ok(Self {
            contract: std::marker::PhantomData,
            kernel,
            buffers,
            load_stages,
            allocation_time,
            upload_time: Duration::ZERO,
            download_time: Duration::ZERO,
            gpu_time: Duration::ZERO,
            gpu_timed_launches: 0,
            audit,
            capacity,
            group,
            load_time,
            dispatch_time: Duration::ZERO,
            verification_time: Duration::ZERO,
            launches: 0,
        })
    }
    fn evaluate_launch(&mut self, launch: &C::Launch, count: u32) -> Result<BatchResult, String> {
        let launch_size = std::mem::size_of::<C::Launch>();
        for buffer in &mut self.buffers {
            buffer.bytes.fill(0xa5);
        }
        self.buffers[0].bytes[256..256 + launch_size].copy_from_slice(bytes(launch));
        self.buffers[1].bytes[256..528].copy_from_slice(bytes(&BatchResult::EMPTY));
        // SAFETY: reviewed entry, ABI-checked bindings, disjoint initialized records,
        // per-lane audit capacity, zeroed atomic counts and bounded dispatch.
        let timing = unsafe {
            self.kernel.run(
                &mut self.buffers,
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
        let start = Instant::now();
        let used = [
            launch_size,
            272,
            if self.audit { count as usize * 260 } else { 0 },
        ];
        for (buffer, used) in self.buffers.iter().zip(used) {
            if buffer.bytes[..256]
                .iter()
                .chain(buffer.bytes[256 + used..].iter())
                .any(|&b| b != 0xa5)
            {
                return Err("Metal kernel changed a guard or unused output byte".into());
            }
        }
        if &self.buffers[0].bytes[256..256 + launch_size] != bytes(launch) {
            return Err("Metal kernel changed input".into());
        }
        let result: BatchResult = record(&self.buffers[1].bytes[256..528]);
        if self.audit {
            let (mut matches, mut errors) = (0, 0);
            for lane in 0..count {
                let expected = C::candidate(launch, lane);
                let offset = 256 + lane as usize * 260;
                let actual = CandidateResult {
                    status: u32::from_le_bytes(
                        self.buffers[2].bytes[offset..offset + 4]
                            .try_into()
                            .unwrap(),
                    ),
                    bytes: self.buffers[2].bytes[offset + 4..offset + 260]
                        .try_into()
                        .unwrap(),
                };
                if actual.status != expected.status || actual.bytes != expected.bytes {
                    return Err(format!("Metal lane {lane} differs from CPU reference"));
                }
                match expected.status {
                    0 => {}
                    1 => matches += 1,
                    _ => errors += 1,
                }
                if result.matches != 0
                    && result.lane == lane
                    && (result.candidate.status != expected.status
                        || result.candidate.bytes != expected.bytes)
                {
                    return Err("Metal winner differs from CPU reference".into());
                }
            }
            if (result.matches, result.errors) != (matches, errors) {
                return Err("Metal batch counts differ from CPU reference".into());
            }
        }
        if let Some((lane, winner)) = result.winner(count)? {
            let expected = C::candidate(launch, lane);
            if winner.status != expected.status || winner.bytes != expected.bytes {
                return Err("Metal winner differs from CPU reference".into());
            }
        }
        self.verification_time += start.elapsed();
        Ok(result)
    }
}

impl Transport<Shallenge> {
    pub fn evaluate(
        &mut self,
        seed: &BatchSeed,
        target: &[u8; 32],
        username: &[u8],
        start: u64,
        count: u32,
    ) -> Result<BatchResult, String> {
        if count == 0
            || count > self.capacity
            || start.checked_add(u64::from(count) - 1).is_none()
            || username.len() > 32
        {
            return Err("invalid Metal candidate range or username storage".into());
        }
        let mut launch = Launch {
            seed: *seed,
            start,
            count,
            username_len: username.len() as u32,
            target: *target,
            username: [0; 32],
            audit: u32::from(self.audit),
            reserved: 0,
        };
        launch.username[..username.len()].copy_from_slice(username);
        self.evaluate_launch(&launch, count)
    }
}

#[cfg(feature = "ethereum")]
impl Transport<Ethereum> {
    pub fn evaluate(
        &mut self,
        seed: &BatchSeed,
        pattern: &logic::search::vanity::BytePattern,
        start: u64,
        count: u32,
    ) -> Result<BatchResult, String> {
        if count == 0 || count > self.capacity || start.checked_add(u64::from(count) - 1).is_none()
        {
            return Err("invalid Metal candidate range".into());
        }
        let launch = super::ethereum_contract::Launch {
            seed: *seed,
            pattern: *pattern,
            start,
            count,
            audit: u32::from(self.audit),
        };
        self.evaluate_launch(&launch, count)
    }
}

#[cfg(feature = "bitcoin")]
impl Transport<Bitcoin> {
    pub fn evaluate(
        &mut self,
        seed: &BatchSeed,
        pattern: &logic::search::vanity::BytePattern,
        start: u64,
        count: u32,
    ) -> Result<BatchResult, String> {
        if count == 0 || count > self.capacity || start.checked_add(u64::from(count) - 1).is_none()
        {
            return Err("invalid Metal candidate range".into());
        }
        let launch = super::bitcoin_contract::Launch {
            seed: *seed,
            pattern: *pattern,
            start,
            count,
            audit: u32::from(self.audit),
        };
        self.evaluate_launch(&launch, count)
    }
}
