//! CUDA transport for the shared cryptographic candidate evaluators.
use cust::{
    context::{Context, CurrentContext, ResourceLimit},
    launch,
    memory::{CopyDestination, DeviceBuffer, DeviceCopy},
    module::Module,
    stream::{Stream, StreamFlags},
};
use logic::{device_search::*, hex_pattern::HexPattern};
use vanity_miner::device_search::{DeviceSearch, Request};
use zeroize::{Zeroize, Zeroizing};

#[repr(transparent)]
#[derive(Clone, Copy)]
struct Record<T>(T);
// Only fixed-layout records of integer fields and arrays are eligible. No
// pointers, references, enum discriminants, or uninitialized padding are copied.
trait Abi: Copy {}
impl Abi for HexPattern {}
impl Abi for CandidateResult {}
#[cfg(feature = "p256-public-key")]
impl Abi for P256PublicRequest {}
#[cfg(feature = "p256-signature")]
impl Abi for P256SignatureRequest {}
#[cfg(feature = "rsa-pss")]
impl Abi for RsaPssRequest {}
#[cfg(feature = "rsa-modulus")]
impl Abi for RsaModulusRequest {}
unsafe impl<T: Abi> DeviceCopy for Record<T> {}
impl<T: Zeroize> Zeroize for Record<T> {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

struct SecretBuffer<'a, T: Abi + Zeroize> {
    buffer: DeviceBuffer<Record<T>>,
    zeros: Zeroizing<Vec<Record<T>>>,
    stream: &'a Stream,
}
impl<'a, T: Abi + Zeroize> SecretBuffer<'a, T> {
    fn new(values: &[Record<T>], stream: &'a Stream) -> Result<Self, String> {
        let buffer = DeviceBuffer::from_slice(values).map_err(|e| e.to_string())?;
        let mut zeros = Zeroizing::new(values.to_vec());
        for value in zeros.iter_mut() {
            value.zeroize();
        }
        Ok(Self {
            buffer,
            zeros,
            stream,
        })
    }
    fn clear(&mut self) -> Result<(), String> {
        self.stream.synchronize().map_err(|e| e.to_string())?;
        self.buffer
            .copy_from(&self.zeros[..])
            .map_err(|e| e.to_string())
    }
}
impl<T: Abi + Zeroize> Drop for SecretBuffer<'_, T> {
    fn drop(&mut self) {
        if self.clear().is_err() {
            eprintln!("CUDA buffer erasure failed; the device context will be released.");
        }
    }
}

struct DeviceState {
    // Drop GPU resources before the context. The engine selects this context
    // before each launch and before destroying the state.
    stream: Stream,
    module: Module,
    context: Context,
}
impl Drop for DeviceState {
    fn drop(&mut self) {
        let _ = CurrentContext::set_current(&self.context);
        let _ = self.stream.synchronize();
    }
}

pub struct Engine {
    devices: Vec<DeviceState>,
    next: usize,
}
impl Engine {
    pub fn new(modules: Vec<(Context, Module)>) -> Result<Self, String> {
        if modules.is_empty() {
            return Err("no CUDA devices available for cryptographic search".into());
        }
        let stack = std::env::var("STACK_SIZE")
            .unwrap_or_else(|_| "65536".into())
            .parse::<usize>()
            .map_err(|_| "invalid STACK_SIZE")?;
        let mut devices = Vec::new();
        for (context, module) in modules {
            CurrentContext::set_current(&context).map_err(|e| e.to_string())?;
            CurrentContext::set_resource_limit(ResourceLimit::StackSize, stack)
                .map_err(|e| e.to_string())?;
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| e.to_string())?;
            devices.push(DeviceState {
                stream,
                module,
                context,
            });
        }
        println!(
            "Cryptographic CUDA search: 64 candidates per batch, rotating across {} devices; host verifies every match.",
            devices.len()
        );
        Ok(Self { devices, next: 0 })
    }

    fn launch<T: Abi + Zeroize>(
        state: &DeviceState,
        name: &str,
        request: &T,
        pattern: &HexPattern,
        message: &[u8],
        start: u64,
        count: u32,
    ) -> Result<Vec<CandidateResult>, String> {
        let stream = &state.stream;
        let kernel = state.module.get_function(name).map_err(|e| e.to_string())?;
        let host_request = Zeroizing::new([Record(*request)]);
        let mut request_device = SecretBuffer::new(&host_request[..], stream)?;
        let pattern_device =
            DeviceBuffer::from_slice(&[Record(*pattern)]).map_err(|e| e.to_string())?;
        // A nonempty allocation gives even empty messages a valid base pointer.
        let message_device =
            DeviceBuffer::from_slice(if message.is_empty() { &[0] } else { message })
                .map_err(|e| e.to_string())?;
        let mut host_results = Zeroizing::new(vec![Record(CandidateResult::MISS); count as usize]);
        let mut result_device = SecretBuffer::new(&host_results, stream)?;
        let operation = (|| {
            unsafe {
                launch!(kernel<<<count.div_ceil(32), 32, 0, stream>>>(
                    request_device.buffer.as_device_ptr(), pattern_device.as_device_ptr(),
                    message_device.as_device_ptr(), message.len(), start, count,
                    result_device.buffer.as_device_ptr()
                ))
                .map_err(|e| e.to_string())?;
            }
            stream.synchronize().map_err(|e| e.to_string())?;
            result_device
                .buffer
                .copy_to(&mut host_results[..])
                .map_err(|e| e.to_string())
        })();
        // Attempt both erasures on every path, including failed launches.
        let cleared_request = request_device.clear();
        let cleared_result = result_device.clear();
        operation?;
        cleared_request?;
        cleared_result?;
        Ok(host_results.iter().map(|record| record.0).collect())
    }
}
impl DeviceSearch for Engine {
    fn evaluate(
        &mut self,
        request: &Request<'_>,
        pattern: &HexPattern,
        message: &[u8],
        start: u64,
        count: u32,
    ) -> Result<Vec<CandidateResult>, String> {
        if count == 0 || count > 64 {
            return Err("invalid CUDA cryptographic batch size".into());
        }
        let state = &self.devices[self.next];
        self.next = (self.next + 1) % self.devices.len();
        CurrentContext::set_current(&state.context).map_err(|e| e.to_string())?;
        match request {
            #[cfg(feature = "p256-public-key")]
            Request::P256Public(request) => Self::launch(
                state,
                "kernel_p256_public_key_vanity",
                *request,
                pattern,
                message,
                start,
                count,
            ),
            #[cfg(feature = "p256-signature")]
            Request::P256Signature(request) => Self::launch(
                state,
                "kernel_p256_signature_vanity",
                *request,
                pattern,
                message,
                start,
                count,
            ),
            #[cfg(feature = "rsa-pss")]
            Request::RsaPss(request) => Self::launch(
                state,
                "kernel_rsa_pss_signature_vanity",
                *request,
                pattern,
                message,
                start,
                count,
            ),
            #[cfg(feature = "rsa-modulus")]
            Request::RsaModulus(request) => Self::launch(
                state,
                "kernel_rsa_modulus_vanity",
                *request,
                pattern,
                message,
                start,
                count,
            ),
        }
    }
}
