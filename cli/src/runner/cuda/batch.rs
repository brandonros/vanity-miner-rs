//! Static inputs, output allocation, and function lookup persist across launches.
use super::buffers::{Message, Records};
use crate::runner::cuda::context::GpuContext;
use cust::{function::Function, launch};
use logic::search::{candidate_result::BatchResult, device_record::DeviceRecord};

pub(crate) struct CandidateBatch<'a, T: DeviceRecord, P: DeviceRecord> {
    gpu: &'a GpuContext,
    kernel: Function<'a>,
    request: Records<'a, T>,
    pattern: Records<'a, P>,
    message: Message<'a>,
    message_len: usize,
    output: Records<'a, BatchResult>,
}

impl<'a, T: DeviceRecord, P: DeviceRecord> CandidateBatch<'a, T, P> {
    pub fn new(
        gpu: &'a GpuContext,
        name: &str,
        request: &T,
        pattern: &P,
        message: &[u8],
    ) -> Result<Self, String> {
        Ok(Self {
            gpu,
            kernel: gpu
                .module()?
                .get_function(name)
                .map_err(|e| e.to_string())?,
            request: Records::from_slice(std::slice::from_ref(request), &gpu.stream)?,
            pattern: Records::from_slice(std::slice::from_ref(pattern), &gpu.stream)?,
            message: Message::new(message, &gpu.stream)?,
            message_len: message.len(),
            output: Records::zeroed(1, &gpu.stream)?,
        })
    }

    pub fn update_pattern(&mut self, pattern: &P) -> Result<(), String> {
        self.pattern.write(std::slice::from_ref(pattern))
    }

    pub fn evaluate(&mut self, start: u64, count: u32) -> Result<BatchResult, String> {
        if !(1..=1_048_576).contains(&count) || start.checked_add(u64::from(count) - 1).is_none() {
            return Err("invalid CUDA candidate range".into());
        }
        self.output.clear_async()?;
        let stream = &self.gpu.stream;
        let kernel = &self.kernel;
        let threads = self.gpu.threads_per_block as u32;
        let blocks = count.div_ceil(threads);
        unsafe {
            launch!(kernel<<<blocks, threads, 0, stream>>>(self.request.pointer(), self.pattern.pointer(), self.message.bytes.as_device_ptr(), self.message_len, start, count, self.output.pointer())).map_err(|e| e.to_string())?;
        }
        stream.synchronize().map_err(|e| e.to_string())?;
        let result = self.output.read(1)?[0];
        result.winner(count)?;
        Ok(result)
    }
}
