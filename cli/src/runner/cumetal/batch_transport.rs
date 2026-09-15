//! Transport for the same RSA/P-256 kernels and host verification used by CUDA.
use super::driver::Buffer;
use super::{Driver, Error, Module};
use logic::search::{
    candidate_result::{BatchResult, CandidateResult},
    hex_pattern::HexPattern,
};
use std::rc::Rc;
use zeroize::Zeroizing;

use logic::search::device_record::DeviceRecord as Abi;
fn bytes<T: Abi>(value: &T) -> &[u8] {
    // SAFETY: DeviceRecord guarantees padding-free integer records.
    unsafe { std::slice::from_raw_parts((value as *const T).cast(), std::mem::size_of::<T>()) }
}
struct SecretBuffer(Buffer);
impl Drop for SecretBuffer {
    fn drop(&mut self) {
        if let Err(error) = self.0.clear() {
            eprintln!("CuMetal buffer erasure failed: {error}");
        }
    }
}
pub(crate) struct Engine<'a> {
    pub(crate) driver: &'a Rc<Driver>,
    pub(crate) module: Module,
    pub(crate) verify: bool,
}
impl Engine<'_> {
    pub(crate) fn evaluate<T: Abi>(
        &mut self,
        request: &T,
        pattern: &HexPattern,
        message: &[u8],
        start: u64,
        count: u32,
        reference: impl Fn(u64) -> CandidateResult,
    ) -> Result<BatchResult, String> {
        if count == 0 || count > 64 || start.checked_add(count as u64 - 1).is_none() {
            return Err("invalid CuMetal cryptographic batch range".into());
        }
        let result = (|| -> Result<BatchResult, Error> {
            let request_device = SecretBuffer(self.driver.buffer(bytes(request))?);
            let pattern_device = self.driver.buffer(bytes(pattern))?;
            let message_device = SecretBuffer(self.driver.buffer(message)?);
            let output = SecretBuffer(self.driver.buffer(bytes(&BatchResult::EMPTY))?);
            self.module.launch(
                &mut [
                    request_device.0.pointer(),
                    pattern_device.pointer(),
                    message_device.0.pointer(),
                    message.len() as u64,
                    start,
                    count as u64,
                    output.0.pointer(),
                ],
                count.div_ceil(32),
                32,
            )?;
            let raw = Zeroizing::new(output.0.read()?);
            // SAFETY: exact-sized BatchResult of integers; every bit pattern is valid.
            let result = unsafe { std::ptr::read_unaligned(raw.as_ptr().cast::<BatchResult>()) };
            // Check input guards as well as the result buffer's guards.
            let _request = Zeroizing::new(request_device.0.read()?);
            let _message = Zeroizing::new(message_device.0.read()?);
            pattern_device.read()?;
            if self.verify {
                let mut matches = 0;
                let mut errors = 0;
                for lane in 0..count {
                    let candidate = Zeroizing::new(reference(start + lane as u64));
                    match candidate.status {
                        0 => {}
                        1 => matches += 1,
                        _ => errors += 1,
                    }
                    if result.matches > 0
                        && lane == result.lane
                        && (result.candidate.status != candidate.status
                            || result.candidate.bytes != candidate.bytes)
                    {
                        return Err("CuMetal winner differs from CPU reference".into());
                    }
                }
                if result.matches != matches || result.errors != errors {
                    return Err("CuMetal batch counts differ from CPU reference".into());
                }
            }
            result.winner(count)?;
            Ok(result)
        })()
        .map_err(|e| e.to_string());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transport_records_have_no_padding_and_batch_result_round_trips() {
        assert_eq!(std::mem::size_of::<HexPattern>(), 516);
        assert_eq!(std::mem::size_of::<BatchResult>(), 272);
        let original = BatchResult {
            matches: 3,
            errors: 0,
            lane: 7,
            candidate: CandidateResult::matched(&[0x42; 32]),
        };
        let copied =
            unsafe { std::ptr::read_unaligned(bytes(&original).as_ptr().cast::<BatchResult>()) };
        assert_eq!(copied.matches, 3);
        assert_eq!(copied.lane, 7);
        assert_eq!(copied.candidate.bytes, original.candidate.bytes);
        assert!(copied.winner(8).unwrap().is_some());
    }
}
