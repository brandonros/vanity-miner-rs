//! Shared GPU candidate reconstruction, verification, and output.
use crate::runner::{batches, session::SearchControl};
use logic::search::{candidate_result::BatchResult, vanity::BytePattern, xoroshiro::BatchSeed};

#[cfg(any(feature = "gpu", feature = "cumetal"))]
pub const ENTRY: &str = "kernel_solana_vanity";

pub fn search(
    prefix: &str,
    suffix: &str,
    seed: Option<u64>,
    control: &SearchControl,
    mut evaluate: impl FnMut(&BatchSeed, &BytePattern, &[u8], u64, u32) -> Result<BatchResult, String>,
) -> Result<Option<String>, String> {
    let pattern = BytePattern::new(prefix.as_bytes(), suffix.as_bytes())?;
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
    let result = logic::modes::solana::generate_and_check_solana_vanity_key(
        &logic::modes::solana::SolanaVanityKeyRequest {
            prefix,
            suffix,
            rng_seed,
            thread_idx,
        },
    );
    if !result.matches || bytes[..32] != result.private_key || bytes[32..].iter().any(|&b| b != 0) {
        return Err("device solana candidate failed CPU verification".into());
    }
    let address = std::str::from_utf8(&result.encoded_public_key[..result.encoded_len])
        .map_err(|e| e.to_string())?;
    Ok(format!(
        "[solana] private_key={}\n[solana] public_key={}\n[solana] address={}\n[solana] wallet={}",
        hex::encode(result.private_key),
        hex::encode(result.public_key),
        address,
        hex::encode([result.private_key, result.public_key].concat())
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verifies_candidate_and_rejects_corrupted_payload() {
        let seed = BatchSeed { seed: 1, width: 32 };
        let pattern = BytePattern::new(b"", b"").unwrap();
        let result = logic::modes::solana::candidate(&seed, 33, &pattern);
        assert!(verify(&seed, &pattern, 33, &result.bytes).is_ok());
        let mut corrupt = result.bytes;
        corrupt[0] ^= 1;
        assert!(verify(&seed, &pattern, 33, &corrupt).is_err());
        assert!(verify(&seed, &pattern, 34, &result.bytes).is_err());
    }
}
