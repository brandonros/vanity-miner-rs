//! Persistent transport for structured candidate requests and batch results.
use super::{
    Driver, Error,
    driver::{Buffer, Module, record_bytes},
};
use logic::search::{
    candidate_result::{BatchResult, CandidateResult},
    device_record::DeviceRecord,
};
use std::rc::Rc;
use zeroize::Zeroizing;

fn bytes<T: DeviceRecord>(value: &T) -> &[u8] {
    record_bytes(std::slice::from_ref(value))
}

pub(crate) struct CumetalBatchTransport<'a> {
    driver: &'a Rc<Driver>,
    module: Module,
    verify: bool,
    threads: u32,
    buffers: Option<[Buffer; 4]>,
}
impl<'a> CumetalBatchTransport<'a> {
    pub fn new(driver: &'a Rc<Driver>, module: Module, verify: bool, threads: u32) -> Self {
        Self {
            driver,
            module,
            verify,
            threads,
            buffers: None,
        }
    }

    pub(crate) fn evaluate<R: DeviceRecord, P: DeviceRecord>(
        &mut self,
        request: &R,
        pattern: &P,
        message: &[u8],
        start: u64,
        count: u32,
        reference: impl Fn(u64) -> CandidateResult,
    ) -> Result<BatchResult, String> {
        if count == 0 || count > 1_048_576 || start.checked_add(u64::from(count) - 1).is_none() {
            return Err("invalid CuMetal candidate range".into());
        }
        (|| -> Result<BatchResult, Error> {
            let inputs = [
                bytes(request),
                bytes(pattern),
                message,
                bytes(&BatchResult::EMPTY),
            ];
            if self.buffers.is_none() {
                self.buffers = Some([
                    self.driver.buffer(inputs[0])?,
                    self.driver.buffer(inputs[1])?,
                    self.driver.buffer(inputs[2])?,
                    self.driver.buffer(inputs[3])?,
                ]);
            } else {
                for (buffer, input) in self.buffers.as_ref().unwrap().iter().zip(inputs) {
                    buffer.write(input)?;
                }
            }
            let [request, pattern, message_device, output] = self.buffers.as_ref().unwrap();
            self.module.launch(
                &mut [
                    request.pointer(),
                    pattern.pointer(),
                    message_device.pointer(),
                    message.len() as u64,
                    start,
                    u64::from(count),
                    output.pointer(),
                ],
                count.div_ceil(self.threads),
                self.threads,
            )?;
            let result = output.read_records::<BatchResult>(1)?[0];
            for (buffer, expected) in [request, pattern, message_device]
                .into_iter()
                .zip(&inputs[..3])
            {
                let actual = Zeroizing::new(buffer.read()?);
                if actual.as_slice() != *expected {
                    return Err("kernel changed an input buffer".into());
                }
            }
            result.winner(count)?;
            if self.verify {
                let (mut matches, mut errors) = (0, 0);
                for lane in 0..count {
                    let candidate = Zeroizing::new(reference(start + u64::from(lane)));
                    match candidate.status {
                        CandidateResult::STATUS_MISS => {}
                        CandidateResult::STATUS_MATCH => matches += 1,
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
            Ok(result)
        })()
        .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use logic::search::hex_pattern::HexPattern;
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
