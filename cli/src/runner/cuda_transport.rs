//! CUDA transport for the shared cryptographic candidate evaluators.
use cust::{
    launch,
    memory::{CopyDestination, DeviceBuffer, DeviceCopy},
    stream::Stream,
};
use logic::search::device_record::DeviceRecord;
use logic::{search::candidate_result::BatchResult, search::hex_pattern::HexPattern};
use zeroize::{Zeroize, Zeroizing};

#[repr(transparent)]
#[derive(Clone, Copy)]
struct Record<T>(T);
// SAFETY: DeviceRecord guarantees pointer-free, initialized, valid copied bytes.
unsafe impl<T: DeviceRecord> DeviceCopy for Record<T> {}
impl<T: Zeroize> Zeroize for Record<T> {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

struct SecretBuffer<'a, T: DeviceRecord + Zeroize> {
    buffer: DeviceBuffer<Record<T>>,
    zeros: Zeroizing<Vec<Record<T>>>,
    stream: &'a Stream,
}
impl<'a, T: DeviceRecord + Zeroize> SecretBuffer<'a, T> {
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
impl<T: DeviceRecord + Zeroize> Drop for SecretBuffer<'_, T> {
    fn drop(&mut self) {
        if self.clear().is_err() {
            eprintln!("CUDA buffer erasure failed; the device context will be released.");
        }
    }
}

pub struct CudaBatchTransport<'a> {
    gpu: &'a crate::common::GpuContext,
}
impl<'a> CudaBatchTransport<'a> {
    pub fn new(gpu: &'a crate::common::GpuContext) -> Self {
        Self { gpu }
    }
    pub fn evaluate<T: DeviceRecord + Zeroize>(
        &mut self,
        name: &str,
        request: &T,
        pattern: &HexPattern,
        message: &[u8],
        start: u64,
        count: u32,
    ) -> Result<BatchResult, String> {
        if count == 0 || count > 1_048_576 {
            return Err("invalid CUDA batch size".into());
        }
        let state = self.gpu;
        let stream = &state.stream;
        let threads = state.threads_per_block as u32;
        let blocks = count.div_ceil(threads);
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
                launch!(kernel<<<blocks, threads, 0, stream>>>(
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
