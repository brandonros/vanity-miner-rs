//! Shared GPU candidate reconstruction, verification, and output.
use crate::runner::{batches, session::SearchControl};
use logic::search::{candidate_result::BatchResult, vanity::BytePattern, xoroshiro::BatchSeed};

pub fn search(
    prefix: &str,
    suffix: &str,
    seed: Option<u64>,
    control: &SearchControl,
    mut evaluate: impl FnMut(&BatchSeed, &BytePattern, &[u8], u64, u32) -> Result<BatchResult, String>,
) -> Result<Option<String>, String> {
    let pattern = BytePattern::new(
        &hex::decode(prefix).map_err(|e| e.to_string())?,
        &hex::decode(suffix).map_err(|e| e.to_string())?,
    )?;
    let request = BatchSeed {
        seed: seed.unwrap_or_else(rand::random),
        width: u64::from(control.batch_size()),
    };
    let output = batches::search(
        |start, count| evaluate(&request, &pattern, &[], start, count),
        u64::MAX,
        control,
        |counter, bytes| verify(&request, &pattern, counter, bytes).map(Some),
    )?;
    if control.exit_on_first_match()
        && let Some(record) = &output
    {
        crate::runner::progress::print_verified(control, record.clone())?;
    }
    Ok(output)
}

fn verify(
    seed: &BatchSeed,
    pattern: &BytePattern,
    counter: u64,
    bytes: &[u8; 256],
) -> Result<String, String> {
    let (rng_seed, thread_idx) = seed.position(counter).ok_or("invalid candidate index")?;
    let (prefix, suffix) = pattern.parts().ok_or("invalid address pattern")?;
    let result = logic::modes::ethereum::generate_and_check_ethereum_vanity_key(
        &logic::modes::ethereum::EthereumVanityKeyRequest {
            prefix,
            suffix,
            rng_seed,
            thread_idx,
        },
    );
    if !result.matches || bytes[..32] != result.private_key || bytes[32..].iter().any(|&b| b != 0) {
        return Err("device ethereum candidate failed CPU verification".into());
    }
    Ok(format!(
        "[ethereum] private_key={}\n[ethereum] public_key={}\n[ethereum] address=0x{}",
        hex::encode(result.private_key),
        hex::encode(result.public_key),
        hex::encode(result.address)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verifies_candidate_and_rejects_corrupted_payload() {
        let seed = BatchSeed { seed: 1, width: 32 };
        let pattern = BytePattern::new(b"", b"").unwrap();
        let result = logic::modes::ethereum::candidate(&seed, 33, &pattern);
        assert!(verify(&seed, &pattern, 33, &result.bytes).is_ok());
        let mut corrupt = result.bytes;
        corrupt[0] ^= 1;
        assert!(verify(&seed, &pattern, 33, &corrupt).is_err());
        assert!(verify(&seed, &pattern, 34, &result.bytes).is_err());
    }
}
