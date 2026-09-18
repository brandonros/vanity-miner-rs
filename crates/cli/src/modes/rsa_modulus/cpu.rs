//! CPU workers invoke the same no_std mining function as the GPU wrapper.
use super::*;
use logic::modes::rsa_modulus::{self as mining, SearchConfig};

pub(super) fn construct_worker(
    config: &SearchConfig,
    pattern: &HexPattern,
    control: &SearchControl,
) -> Result<Option<String>, String> {
    let _stop_peers = control.cancel_on_exit();
    while !control.stopped() {
        let Some(ids) = control.reserve_batch(u64::from(control.batch_size())) else {
            break;
        };
        for id in ids {
            if control.stopped() {
                break;
            }
            let result = Zeroizing::new(mining::rsa_modulus(config, id, pattern));
            control.add_tested(1);
            match result.status {
                0 => (),
                1 => {
                    let pair = Zeroizing::new(mining::Pair {
                        p: result.bytes[..128].try_into().unwrap(),
                        q: result.bytes[128..].try_into().unwrap(),
                        id,
                    });
                    let output = pipeline::verify_pair(config, pattern, &pair)?;
                    if control.claim_verified_winner() {
                        return Ok(Some(output));
                    }
                    return Ok(None);
                }
                _ => return Err("RSA candidate evaluation failed".into()),
            }
        }
    }
    Ok(None)
}

#[cfg(not(any(feature = "gpu", feature = "cumetal", feature = "metal")))]
pub fn run(
    args: &crate::modes::rsa_modulus::args::RsaModulusArgs,
    workers: usize,
    stats: std::sync::Arc<crate::runner::progress::GlobalStats>,
    exit_on_first_match: bool,
) -> crate::runner::RunResult {
    use crate::runner::{progress::estimate, session::run_controlled};
    let config = args.config(workers)?;
    estimate(config.validate()?.pattern.constrained_bits() - 2);
    println!("Constructive search restricts every q candidate to the requested modulus pattern.");
    run_controlled(stats, "candidates", exit_on_first_match, |control| {
        super::run_cpu(&config, control).map(|report| report.found)
    })
}

#[cfg(test)]
#[path = "../../../tests/support/rsa_factors.rs"]
mod test_factors;

#[cfg(test)]
mod tests {
    use super::test_factors as factors;
    use super::*;
    #[test]
    fn worker_completes_one_candidate_and_obeys_cancellation() {
        let n = BigUint::from_bytes_be(&factors::P) * BigUint::from_bytes_be(&factors::Q);
        let constraints = ModulusConstraints::new(&hex::encode(n.to_bytes_be()), "").unwrap();
        let mut config = constraints.device_config([42; 32], 0).unwrap();
        config.p_min = factors::P;
        config.p_count[127] = 1;
        config.p_count[..127].fill(0);
        let control = SearchControl::new();
        assert!(
            construct_worker(&config, &constraints.pattern, &control)
                .unwrap()
                .is_some()
        );
        assert_eq!(control.statistics().0, 1);
        let control = SearchControl::new();
        control.cancel();
        assert!(
            construct_worker(&config, &constraints.pattern, &control)
                .unwrap()
                .is_none()
        );
        assert_eq!(control.statistics().0, 0);
    }
}
