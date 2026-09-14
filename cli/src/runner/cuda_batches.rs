//! CUDA transport for the shared cryptographic candidate evaluators.
use cust::{
    launch,
    memory::{CopyDestination, DeviceBuffer, DeviceCopy},
    stream::Stream,
};
#[cfg(feature = "p256-public-key")]
use logic::p256_public_key_vanity::P256PublicRequest;
#[cfg(feature = "p256-signature")]
use logic::p256_signature_vanity::P256SignatureRequest;
#[cfg(feature = "rsa-modulus")]
use logic::rsa_modulus_vanity::RsaModulusRequest;
#[cfg(feature = "rsa-pss")]
use logic::rsa_pss_signature_vanity::RsaPssRequest;
use logic::{candidate_result::BatchResult, hex_pattern::HexPattern};
use zeroize::{Zeroize, Zeroizing};

#[repr(transparent)]
#[derive(Clone, Copy)]
struct Record<T>(T);
// Only fixed-layout records of integer fields and arrays are eligible. No
// pointers, references, enum discriminants, or uninitialized padding are copied.
trait Abi: Copy {}
impl Abi for HexPattern {}
impl Abi for BatchResult {}
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

pub struct Engine<'a> {
    gpu: &'a crate::common::GpuContext,
}
impl<'a> Engine<'a> {
    pub fn new(gpu: &'a crate::common::GpuContext) -> Self {
        Self { gpu }
    }
    fn launch<T: Abi + Zeroize>(
        state: &crate::common::GpuContext,
        name: &str,
        request: &T,
        pattern: &HexPattern,
        message: &[u8],
        start: u64,
        count: u32,
    ) -> Result<BatchResult, String> {
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
        let mut host_results = Zeroizing::new([Record(BatchResult::EMPTY)]);
        let mut result_device = SecretBuffer::new(&host_results[..], stream)?;
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
        Ok(host_results[0].0)
    }
}
impl Engine<'_> {
    #[cfg(feature = "p256-public-key")]
    pub fn p256_public(
        &mut self,
        request: &P256PublicRequest,
        pattern: &HexPattern,
        message: &[u8],
        start: u64,
        count: u32,
    ) -> Result<BatchResult, String> {
        if count == 0 || count > 64 {
            return Err("invalid CUDA cryptographic batch size".into());
        }
        let state = self.gpu;
        Self::launch(
            state,
            "kernel_p256_public_key_vanity",
            request,
            pattern,
            message,
            start,
            count,
        )
    }
    #[cfg(feature = "p256-signature")]
    pub fn p256_signature(
        &mut self,
        request: &P256SignatureRequest,
        pattern: &HexPattern,
        message: &[u8],
        start: u64,
        count: u32,
    ) -> Result<BatchResult, String> {
        if count == 0 || count > 64 {
            return Err("invalid CUDA cryptographic batch size".into());
        }
        let state = self.gpu;
        Self::launch(
            state,
            "kernel_p256_signature_vanity",
            request,
            pattern,
            message,
            start,
            count,
        )
    }
    #[cfg(feature = "rsa-pss")]
    pub fn rsa_pss(
        &mut self,
        request: &RsaPssRequest,
        pattern: &HexPattern,
        message: &[u8],
        start: u64,
        count: u32,
    ) -> Result<BatchResult, String> {
        if count == 0 || count > 64 {
            return Err("invalid CUDA cryptographic batch size".into());
        }
        let state = self.gpu;
        Self::launch(
            state,
            "kernel_rsa_pss_signature_vanity",
            request,
            pattern,
            message,
            start,
            count,
        )
    }
    #[cfg(feature = "rsa-modulus")]
    pub fn rsa_modulus(
        &mut self,
        request: &RsaModulusRequest,
        pattern: &HexPattern,
        message: &[u8],
        start: u64,
        count: u32,
    ) -> Result<BatchResult, String> {
        if count == 0 || count > 64 {
            return Err("invalid CUDA cryptographic batch size".into());
        }
        let state = self.gpu;
        Self::launch(
            state,
            "kernel_rsa_modulus_vanity",
            request,
            pattern,
            message,
            start,
            count,
        )
    }
}
