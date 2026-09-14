use std::error::Error;
use vanity_miner::self_test_suite::{self, Kind, Outcome};

#[cfg(not(feature = "gpu"))]
pub mod cpu {
    use super::*;
    pub fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut results = [0; logic::SELF_TEST_NUM_CHECKS];
        logic::run_self_test(&mut results);
        self_test_suite::run("CPU", |case| {
            match case.kind {
                Kind::Probe => return Ok(Outcome::Skipped("requires GPU launch")),
                Kind::Legacy(slot) => {
                    if results[slot] != 1 {
                        return Err("known-answer mismatch".into());
                    }
                }
                #[cfg(feature = "p256-public-key")]
                Kind::P256Public => vanity_miner::device_self_test::p256_public(
                    &mut vanity_miner::test_support::p256_public,
                )?,
                #[cfg(feature = "p256-signature")]
                Kind::P256Signature => vanity_miner::device_self_test::p256_signature(
                    &mut vanity_miner::test_support::p256_signature,
                )?,
                #[cfg(feature = "rsa-pss")]
                Kind::RsaPss => vanity_miner::device_self_test::rsa_pss(
                    &mut vanity_miner::test_support::rsa_pss,
                )?,
                #[cfg(feature = "rsa-modulus")]
                Kind::RsaModulus => vanity_miner::device_self_test::rsa_modulus(
                    &mut vanity_miner::test_support::rsa_modulus,
                )?,
                #[allow(unreachable_patterns)]
                _ => return Ok(Outcome::Skipped("mode feature disabled")),
            }
            Ok(Outcome::Passed)
        })
        .map_err(Into::into)
    }
}
#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use crate::common::GpuContext;
    use cust::{
        launch,
        memory::{CopyDestination, DeviceBuffer},
    };
    pub fn run(ordinal: usize, gpu: &GpuContext) -> Result<(), Box<dyn Error + Send + Sync>> {
        #[cfg(feature = "crypto-cli")]
        let mut engine = crate::runner::cuda_batches::Engine::new(gpu);
        self_test_suite::run(&format!("CUDA {ordinal}"), |case| {
            match case.kind {
                Kind::Probe | Kind::Legacy(_) => {
                    let stream = &gpu.stream;
                    let mut results = [0u32; logic::SELF_TEST_NUM_CHECKS];
                    let device =
                        DeviceBuffer::<u32>::zeroed(results.len()).map_err(|e| e.to_string())?;
                    let kernel = gpu
                        .module
                        .get_function(case.kernel)
                        .map_err(|e| e.to_string())?;
                    unsafe { launch!(kernel<<<1u32, 1u32, 0, stream>>>(device.as_device_ptr())) }
                        .map_err(|e| e.to_string())?;
                    stream.synchronize().map_err(|e| e.to_string())?;
                    device.copy_to(&mut results).map_err(|e| e.to_string())?;
                    let slot = match case.kind {
                        Kind::Legacy(slot) => slot,
                        _ => 0,
                    };
                    if results[slot] != 1 {
                        return Err("known-answer mismatch".into());
                    }
                }
                #[cfg(feature = "p256-public-key")]
                Kind::P256Public => {
                    vanity_miner::device_self_test::p256_public(&mut |r, p, m, s, c| {
                        engine.p256_public(r, p, m, s, c)
                    })?
                }
                #[cfg(feature = "p256-signature")]
                Kind::P256Signature => {
                    vanity_miner::device_self_test::p256_signature(&mut |r, p, m, s, c| {
                        engine.p256_signature(r, p, m, s, c)
                    })?
                }
                #[cfg(feature = "rsa-pss")]
                Kind::RsaPss => vanity_miner::device_self_test::rsa_pss(&mut |r, p, m, s, c| {
                    engine.rsa_pss(r, p, m, s, c)
                })?,
                #[cfg(feature = "rsa-modulus")]
                Kind::RsaModulus => {
                    vanity_miner::device_self_test::rsa_modulus(&mut |r, p, m, s, c| {
                        engine.rsa_modulus(r, p, m, s, c)
                    })?
                }
                #[allow(unreachable_patterns)]
                _ => return Ok(Outcome::Skipped("mode feature disabled")),
            }
            Ok(Outcome::Passed)
        })
        .map_err(Into::into)
    }
}
