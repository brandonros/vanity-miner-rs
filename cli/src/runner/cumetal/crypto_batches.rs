//! Transport for the same RSA/P-256 kernels and host verification used by CUDA.
use super::driver::Buffer;
use super::{CumetalRunner, Driver, Error, Module};
use crate::{
    args::Command,
    common::{GlobalStats, search_session::run_controlled},
};
use logic::search::{
    candidate_result::{BatchResult, CandidateResult},
    hex_pattern::HexPattern,
};
use std::{rc::Rc, sync::Arc};
use zeroize::Zeroizing;

// Only padding-free repr(C) integer/byte-array records implement this private trait.
trait Abi: Copy {}
impl Abi for HexPattern {}
impl Abi for BatchResult {}
#[cfg(feature = "p256-public-key")]
use logic::modes::p256_public_key_vanity::P256PublicRequest;
#[cfg(feature = "p256-public-key")]
impl Abi for P256PublicRequest {}
#[cfg(feature = "p256-signature")]
use logic::modes::p256_signature_vanity::P256SignatureRequest;
#[cfg(feature = "p256-signature")]
impl Abi for P256SignatureRequest {}
#[cfg(feature = "rsa-pss")]
use logic::modes::rsa_pss_signature_vanity::RsaPssRequest;
#[cfg(feature = "rsa-pss")]
impl Abi for RsaPssRequest {}
#[cfg(feature = "rsa-modulus")]
use logic::modes::rsa_modulus_vanity::RsaModulusRequest;
#[cfg(feature = "rsa-modulus")]
impl Abi for RsaModulusRequest {}
fn bytes<T: Abi>(value: &T) -> &[u8] {
    // SAFETY: Abi is private and implemented only for padding-free integer records.
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
struct Engine<'a> {
    driver: &'a Rc<Driver>,
    module: Module,
    verify: bool,
    remaining: Option<u64>,
    exhausted: bool,
}
impl Engine<'_> {
    fn evaluate<T: Abi>(
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
        if self.remaining == Some(0) {
            self.exhausted = true;
            return Err("CuMetal batch limit reached".into());
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
        if let Some(remaining) = &mut self.remaining {
            *remaining -= 1;
        }
        result
    }
}
impl CumetalRunner {
    pub(super) fn crypto_search(
        &self,
        command: &Command,
        driver: &Rc<Driver>,
        stats: Arc<GlobalStats>,
    ) -> Option<Result<(), Error>> {
        match command {
            #[cfg(feature = "p256-public-key")]
            Command::P256PublicKeyVanity(args) => Some((|| -> Result<(), Error> {
                let config = args.config(1);
                let module = self.module(driver, "kernel_p256_public_key_vanity")?;
                let mut engine = Engine {
                    driver,
                    module,
                    verify: self.options.verify,
                    remaining: self.options.batches,
                    exhausted: false,
                };
                run_controlled(stats, "keys tested", |control| {
                    if engine.remaining == Some(0) {
                        return Ok(false);
                    }
                    let report = vanity_miner::p256_public::run_device(
                        &config,
                        control,
                        &mut |r, p, m, start, count| {
                            engine.evaluate(r, p, m, start, count, |counter| {
                                logic::modes::p256_public_key_vanity::p256_public(r, counter, p)
                            })
                        },
                    );
                    if engine.exhausted {
                        Ok(false)
                    } else {
                        report.map(|r| r.found)
                    }
                })
            })()),
            #[cfg(feature = "p256-signature")]
            Command::P256SignatureVanity(args) => Some((|| -> Result<(), Error> {
                let config = args.config(1)?;
                let module = self.module(driver, "kernel_p256_signature_vanity")?;
                let mut engine = Engine {
                    driver,
                    module,
                    verify: self.options.verify,
                    remaining: self.options.batches,
                    exhausted: false,
                };
                run_controlled(
                    stats,
                    if matches!(
                        config.source,
                        vanity_miner::p256_signature::SearchSource::Message { .. }
                    ) {
                        "messages tested"
                    } else {
                        "nonces tested"
                    },
                    |control| {
                        if engine.remaining == Some(0) {
                            return Ok(false);
                        }
                        let report = vanity_miner::p256_signature::run_device(
                            &config,
                            control,
                            &mut |r, p, m, start, count| {
                                engine.evaluate(r, p, m, start, count, |counter| {
                                    logic::modes::p256_signature_vanity::p256_signature(
                                        r, m, counter, p,
                                    )
                                })
                            },
                        );
                        if engine.exhausted {
                            Ok(false)
                        } else {
                            report.map(|r| r.found)
                        }
                    },
                )
            })()),
            #[cfg(feature = "rsa-pss")]
            Command::RsaPssSignatureVanity(args) => Some((|| -> Result<(), Error> {
                let config = args.config(1)?;
                let module = self.module(driver, "kernel_rsa_pss_signature_vanity")?;
                let mut engine = Engine {
                    driver,
                    module,
                    verify: self.options.verify,
                    remaining: self.options.batches,
                    exhausted: false,
                };
                run_controlled(
                    stats,
                    if matches!(
                        config.source,
                        vanity_miner::rsa_pss_search::PssSource::Salt { .. }
                    ) {
                        "salts tested"
                    } else {
                        "messages tested"
                    },
                    |control| {
                        if engine.remaining == Some(0) {
                            return Ok(false);
                        }
                        let report = vanity_miner::rsa_pss_search::run_device(
                            &config,
                            control,
                            &mut |r, p, m, start, count| {
                                engine.evaluate(r, p, m, start, count, |counter| {
                                    logic::modes::rsa_pss_signature_vanity::rsa_pss(
                                        r, m, counter, p,
                                    )
                                })
                            },
                        );
                        if engine.exhausted {
                            Ok(false)
                        } else {
                            report.map(|r| r.found)
                        }
                    },
                )
            })()),
            #[cfg(feature = "rsa-modulus")]
            Command::RsaModulusVanity(args) => Some((|| -> Result<(), Error> {
                let config = args.config(1)?;
                let module = self.module(driver, "kernel_rsa_modulus_vanity")?;
                let mut engine = Engine {
                    driver,
                    module,
                    verify: self.options.verify,
                    remaining: self.options.batches,
                    exhausted: false,
                };
                run_controlled(stats, "q candidates tested", |control| {
                    if engine.remaining == Some(0) {
                        return Ok(false);
                    }
                    let report = vanity_miner::rsa_modulus::run_device(
                        &config,
                        control,
                        &mut |r, p, m, start, count| {
                            engine.evaluate(r, p, m, start, count, |counter| {
                                logic::modes::rsa_modulus_vanity::rsa_modulus(r, counter, p)
                            })
                        },
                    );
                    if engine.exhausted {
                        Ok(false)
                    } else {
                        report.map(|r| r.found)
                    }
                })
            })()),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transport_records_have_no_padding_and_batch_result_round_trips() {
        assert_eq!(std::mem::size_of::<HexPattern>(), 516);
        assert_eq!(std::mem::size_of::<BatchResult>(), 272);
        #[cfg(feature = "p256-public-key")]
        assert_eq!(std::mem::size_of::<P256PublicRequest>(), 48);
        #[cfg(feature = "p256-signature")]
        assert_eq!(std::mem::size_of::<P256SignatureRequest>(), 168);
        #[cfg(feature = "rsa-pss")]
        assert_eq!(std::mem::size_of::<RsaPssRequest>(), 920);
        #[cfg(feature = "rsa-modulus")]
        assert_eq!(std::mem::size_of::<RsaModulusRequest>(), 512);
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
