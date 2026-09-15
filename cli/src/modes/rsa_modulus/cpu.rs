use super::*;

pub(super) fn construct_worker(
    constraints: &ModulusConstraints,
    control: &SearchControl,
) -> Result<Option<RsaPrivateKey>, String> {
    let _stop_peers = control.cancel_on_exit();
    let zero = BigUint::from(0u8);
    let one = BigUint::from(1u8);
    let e = BigUint::from(65537u32);
    while !control.stopped() {
        let p = Zeroizing::new(constraints.random_p_candidate());
        if (&*p - &one) % &e == zero {
            continue;
        }
        let Some(progression) = constraints.progression(&p) else {
            continue;
        };
        if !probably_prime(&p, 32) {
            continue;
        }
        let start = Zeroizing::new(OsRng.gen_biguint_below(&progression.count));
        // Each p gets a securely randomized start and a nonrepeating progression.
        // Exhaust a small interval once, without retesting the same q.
        for counter in 0..progression.search_budget() {
            if control.stopped() {
                return Ok(None);
            }
            let index = Zeroizing::new((&*start + BigUint::from(counter)) % &*progression.count);
            let q = Zeroizing::new(&*progression.first + &*index * &progression.stride);
            control.add_tested(1);
            if (&*q - &one) % &e == zero
                || !sufficiently_separated(&p, &q)
                || !probably_prime(&q, 32)
            {
                continue;
            }
            let mut key = RsaPrivateKey::from_p_q((*p).clone(), (*q).clone(), e.clone())
                .map_err(|_| "RSA key construction failed")?;
            validate_rsa2048(&mut key)?;
            let modulus = fixed_bytes::<256>(key.n())?;
            if !constraints.pattern.matches(modulus.as_ref()) {
                return Err("constructed RSA modulus failed the pattern".into());
            }
            // Independently verify a blinded private/public operation before export.
            let digest = Sha256::digest(b"vanity-miner RSA key consistency check");
            let signature = key
                .sign_with_rng(&mut OsRng, Pss::new_blinded::<Sha256>(), &digest)
                .map_err(|_| "RSA consistency signing failed")?;
            key.to_public_key()
                .verify(Pss::new::<Sha256>(), &digest, &signature)
                .map_err(|_| "RSA consistency verification failed")?;
            if control.claim_verified_winner() {
                return Ok(Some(key));
            }
            return Ok(None);
        }
    }
    Ok(None)
}

#[cfg(not(any(feature = "gpu", feature = "cumetal")))]
pub fn run(
    args: &crate::modes::rsa_modulus::args::RsaModulusArgs,
    workers: usize,
    stats: std::sync::Arc<crate::runner::progress::GlobalStats>,
) -> crate::runner::RunResult {
    use crate::runner::{progress::estimate, session::run_controlled};
    let config = args.config(workers)?;
    estimate(config.validate()?.pattern.constrained_bits() - 2);
    println!("Constructive search restricts every q candidate to the requested modulus pattern.");
    run_controlled(stats, "q candidates", |control| {
        crate::modes::rsa_modulus::run_cpu(&config, control).map(|report| report.found)
    })
}
