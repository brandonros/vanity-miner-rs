use crate::runner::cumetal::{Error, address_transport::Expected};

pub fn print_payloads(output: &[Vec<u8>]) -> Result<(), Error> {
    {
        let length = output[2]
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(output[2].len());
        println!("private_key={}", hex::encode(&output[0]));
        println!("public_key={}", hex::encode(&output[1]));
        println!("address={}", std::str::from_utf8(&output[2][..length])?);
    };
    Ok(())
}

pub fn inputs(prefix: &str, suffix: &str) -> Result<(Vec<u8>, Vec<u8>), Error> {
    Ok((prefix.as_bytes().to_vec(), suffix.as_bytes().to_vec()))
}

pub fn expected(first: &[u8], second: &[u8], seed: u64, index: usize) -> Result<Expected, Error> {
    Ok({
        let r = logic::modes::solana_vanity::generate_and_check_solana_vanity_key(
            &logic::modes::solana_vanity::SolanaVanityKeyRequest {
                prefix: &first,
                suffix: &second,
                thread_idx: index,
                rng_seed: seed,
            },
        );
        Expected {
            matched: r.matches,
            payloads: vec![
                r.private_key.to_vec(),
                r.public_key.to_vec(),
                r.encoded_public_key.to_vec(),
            ],
        }
    })
}

pub const ENTRY: &str = "kernel_find_solana_vanity_private_key";
pub const PAYLOAD_SIZES: &[usize] = &[32, 32, 64];
