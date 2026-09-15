use super::*;

pub(super) fn construct_device(
    constraints: &ModulusConstraints,
    control: &SearchControl,
    device: &mut EvaluateBatch<'_>,
) -> Result<Option<RsaPrivateKey>, String> {
    use logic::modes::rsa_modulus::RsaModulusRequest;
    if constraints.device_config([0; 32], 0)?.suffix_bits >= 1024 {
        return Err("this legacy batch backend supports suffixes shorter than 128 bytes; use the CUDA RSA pipeline for wider suffixes".into());
    }
    let _stop = control.cancel_on_exit();
    let one = BigUint::from(1u8);
    let e = BigUint::from(65537u32);
    while !control.stopped() {
        // Filter independent p candidates on the device. The CPU only samples
        // an eligible odd interval and independently validates the returned p.
        let max = (&one << 1024usize) - &one;
        let mut min = ceil_div(&constraints.lower, &max).max(&one << 1023usize);
        if &min % 2u8 == BigUint::from(0u8) {
            min += &one;
        }
        let candidates = (&max - &min) / 2u8 + &one;
        let p_count = if candidates >= BigUint::from(control.batch_size()) {
            control.batch_size()
        } else {
            candidates
                .to_bytes_be()
                .iter()
                .fold(0u32, |n, b| (n << 8) | u32::from(*b))
        };
        let offset =
            Zeroizing::new(OsRng.gen_biguint_below(&(&candidates - BigUint::from(p_count - 1))));
        let first_p = Zeroizing::new(&min + &*offset * 2u8);
        let upper_p = Zeroizing::new(&*first_p + BigUint::from(p_count - 1) * 2u8);
        let p_request = Zeroizing::new(RsaModulusRequest {
            stage: 1,
            reserved: 0,
            p: [0; 128],
            first: *fixed_bytes(&first_p)?,
            stride: *fixed_bytes(&BigUint::from(2u8))?,
            upper: *fixed_bytes(&upper_p)?,
        });
        if !control.reserve_device_launch() {
            return Ok(None);
        }
        let results = Zeroizing::new(device(&p_request, &constraints.pattern, &[], 0, p_count)?);
        let winner = results.winner(p_count)?;
        control.add_tested(u64::from(p_count));
        if control.stopped() {
            return Ok(None);
        }
        let Some((lane, result)) = winner else {
            continue;
        };
        let p = Zeroizing::new(&*first_p + BigUint::from(lane) * 2u8);
        if result.bytes[..128] != fixed_bytes::<128>(&p)?[..] {
            return Err("device RSA p factor failed reconstruction".into());
        }
        if (&*p - &one) % &e == BigUint::from(0u8)
            || !crate::modes::rsa_keys::strong_probable_prime(&p)
        {
            continue;
        }
        let Some(progression) = constraints.progression(&p) else {
            continue;
        };
        let budget = progression.search_budget();
        // Choose a start with room for the selected interval, even for one q.
        let start = Zeroizing::new(
            OsRng.gen_biguint_below(&(&*progression.count - BigUint::from(budget - 1))),
        );
        let first = Zeroizing::new(&*progression.first + &*start * &progression.stride);
        let upper = Zeroizing::new(&*first + BigUint::from(budget - 1) * &progression.stride);
        let request = Zeroizing::new(RsaModulusRequest {
            stage: 0,
            reserved: 0,
            p: *fixed_bytes(&p)?,
            first: *fixed_bytes(&first)?,
            stride: *fixed_bytes(&progression.stride)?,
            upper: *fixed_bytes(&upper)?,
        });
        for start in (0..budget).step_by(control.batch_size() as usize) {
            if control.stopped() {
                return Ok(None);
            }
            let count = (budget - start).min(u64::from(control.batch_size())) as u32;
            if !control.reserve_device_launch() {
                return Ok(None);
            }
            let results =
                Zeroizing::new(device(&request, &constraints.pattern, &[], start, count)?);
            let winner = results.winner(count)?;
            control.add_tested(u64::from(count));
            if let Some((lane, result)) = winner {
                if control.stopped() {
                    return Ok(None);
                }
                let q = Zeroizing::new(
                    &*first + BigUint::from(start + lane as u64) * &progression.stride,
                );
                if result.bytes[..128] != fixed_bytes::<128>(&q)?[..] {
                    return Err("device RSA factor failed reconstruction".into());
                }
                if !sufficiently_separated(&p, &q)
                    || !crate::modes::rsa_keys::strong_probable_prime(&q)
                {
                    continue;
                }
                let mut key = RsaPrivateKey::from_p_q((*p).clone(), (*q).clone(), e.clone())
                    .map_err(|_| "device RSA key construction failed")?;
                validate_rsa2048(&mut key)?;
                if !constraints
                    .pattern
                    .matches(&fixed_bytes::<256>(key.n())?[..])
                {
                    return Err("device RSA modulus failed pattern verification".into());
                }
                let digest = Sha256::digest(b"vanity-miner RSA key consistency check");
                let signature = key
                    .sign_with_rng(&mut OsRng, Pss::new_blinded::<Sha256>(), &digest)
                    .map_err(|_| "device RSA consistency signing failed")?;
                key.to_public_key()
                    .verify(Pss::new::<Sha256>(), &digest, &signature)
                    .map_err(|_| "device RSA consistency verification failed")?;
                return Ok(control.claim_verified_winner().then_some(key));
            }
        }
    }
    Ok(None)
}
