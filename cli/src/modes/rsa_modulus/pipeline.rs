//! Host setup and final verification for continuous device-owned RSA searches.
use super::*;
use logic::modes::rsa_modulus::{self as device_logic, Counts, Pair, SearchConfig};
use rand::RngCore;

pub fn run(
    search: &ModulusSearch,
    control: &SearchControl,
    stages: &StageStats,
    steps: u32,
    mut cycle: impl FnMut(
        &SearchConfig,
        &HexPattern,
        u64,
        u32,
    ) -> Result<(Counts, Zeroizing<Vec<Pair>>), String>,
) -> Result<(), String> {
    let constraints = search.validate()?;
    let mut seed = Zeroizing::new([0; 32]);
    OsRng
        .try_fill_bytes(seed.as_mut())
        .map_err(|_| "OS cryptographic entropy unavailable")?;
    let config = Zeroizing::new(constraints.device_config(*seed, 0)?);
    let capacity = control.batch_size();
    let work = device_logic::launch_work(capacity, steps)?;
    // IDs are start + step * capacity + lane, so id % capacity identifies the
    // owning slot. Track retirement without retaining exported private factors.
    let mut retired = vec![None; capacity as usize];
    crate::runner::batches::pump(
        control,
        || {
            let Some(ids) = control.reserve_batch(u64::from(work)) else {
                return Ok(None);
            };
            if !control.reserve_device_launch() {
                return Ok(None);
            }
            let (counts, pairs) = cycle(&config, &constraints.pattern, ids.start, capacity)?;
            counts.validate(capacity, steps)?;
            if pairs.len() != counts.matches as usize {
                return Err("RSA miner returned invalid result count".into());
            }
            if pairs.iter().any(|pair| pair.id >= ids.end) {
                return Err("RSA pipeline returned an unassigned task identifier".into());
            }
            control.add_tested(u64::from(counts.p_tested) + u64::from(counts.q_tested));
            stages.add(
                u64::from(counts.p_tested),
                u64::from(counts.p_accepted),
                u64::from(counts.ranges),
                u64::from(counts.q_tested),
            );
            Ok(Some(pairs))
        },
        |pairs| {
            for pair in pairs.iter() {
                let previous = &mut retired[(pair.id % u64::from(capacity)) as usize];
                if previous.is_some_and(|id| id >= pair.id) {
                    return Err("RSA pipeline attempted to export a retired factor task".into());
                }
                let output = verify_pair(&config, &constraints.pattern, pair)?;
                *previous = Some(pair.id);
                crate::runner::progress::print_verified(control, output)?;
            }
            Ok(())
        },
    )
}

/// The CPU touches factor arithmetic only for a completed candidate pair.
/// Reconstruct p from its PRF input, repeat primality checks with fresh random
/// bases, validate the full key and pattern, then verify a blinded sign/verify.
pub fn verify_pair(
    config: &SearchConfig,
    pattern: &HexPattern,
    pair: &Pair,
) -> Result<String, String> {
    let expected = Zeroizing::new(
        device_logic::generate_p(config, pair.id).ok_or("RSA factor reconstruction failed")?,
    );
    if pair.p != *expected {
        return Err("RSA device p failed reconstruction".into());
    }
    let p = Zeroizing::new(BigUint::from_bytes_be(&pair.p));
    let q = Zeroizing::new(BigUint::from_bytes_be(&pair.q));
    if p.bits() != 1024 || q.bits() != 1024 || !sufficiently_separated(&p, &q) {
        return Err("RSA device pair failed factor bounds or separation".into());
    }
    let mut key = RsaPrivateKey::from_p_q((*p).clone(), (*q).clone(), BigUint::from(65537u32))
        .map_err(|_| "RSA device key construction failed")?;
    validate_rsa2048(&mut key)?;
    if !pattern.matches(fixed_bytes::<256>(key.n())?.as_ref()) {
        return Err("RSA device modulus failed pattern verification".into());
    }
    let digest = Sha256::digest(b"vanity-miner RSA key consistency check");
    let signature = key
        .sign_with_rng(&mut OsRng, Pss::new_blinded::<Sha256>(), &digest)
        .map_err(|_| "RSA consistency signing failed")?;
    key.to_public_key()
        .verify(Pss::new::<Sha256>(), &digest, &signature)
        .map_err(|_| "RSA consistency verification failed")?;
    let private = key
        .to_pkcs8_der()
        .map_err(|_| "RSA private key encoding failed")?;
    Ok(format!(
        "[rsa-modulus] public_key={}\n[rsa-modulus] public_exponent={}\n[rsa-modulus] private_key_pkcs8={}",
        hex::encode(key.n().to_bytes_be()),
        hex::encode(key.e().to_bytes_be()),
        hex::encode(private.as_bytes()),
    ))
}

/// Per-session RSA stage totals shared by all selected devices.
#[derive(Default)]
pub struct StageStats {
    counts: [std::sync::atomic::AtomicU64; 4],
}

impl StageStats {
    pub fn attach(progress: &crate::runner::progress::GlobalStats) -> Result<Arc<Self>, String> {
        let stages = Arc::new(Self::default());
        let display = stages.clone();
        progress.set_details(move |elapsed| display.format(elapsed))?;
        Ok(stages)
    }

    fn add(&self, p: u64, accepted: u64, ranges: u64, q: u64) {
        use std::sync::atomic::Ordering;
        for (counter, amount) in self.counts.iter().zip([p, accepted, ranges, q]) {
            let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                Some(old.saturating_add(amount))
            });
        }
    }

    fn format(&self, elapsed: Duration) -> String {
        use std::sync::atomic::Ordering;
        let [p, accepted, ranges, q] = self.counts.each_ref().map(|v| v.load(Ordering::Relaxed));
        if p == 0 {
            return String::new();
        }
        let seconds = elapsed.as_secs_f64().max(1e-9);
        format!(
            "  RSA stages: {p} p tested ({:.2}/sec), {accepted} probable p, {ranges} nonempty ranges, {q} q tested ({:.2}/sec)",
            p as f64 / seconds,
            q as f64 / seconds
        )
    }
}
