use crate::runner::cumetal::{Error, address_transport::Expected};

pub fn print_payloads(output: &[Vec<u8>]) -> Result<(), Error> {
    {
        let length = u32::from_le_bytes(output[4].as_slice().try_into()?) as usize;
        let address = output[3].get(..length).ok_or("invalid address length")?;
        println!("private_key={}", hex::encode(&output[0]));
        println!("public_key={}", hex::encode(&output[1]));
        println!("hash160={}", hex::encode(&output[2]));
        println!("address={}", std::str::from_utf8(address)?);
    };
    Ok(())
}

pub fn inputs(prefix: &str, suffix: &str) -> Result<(Vec<u8>, Vec<u8>), Error> {
    Ok((prefix.as_bytes().to_vec(), suffix.as_bytes().to_vec()))
}

pub fn expected(first: &[u8], second: &[u8], seed: u64, index: usize) -> Result<Expected, Error> {
    Ok({
        let r = logic::modes::bitcoin_vanity::generate_and_check_bitcoin_vanity_key(
            &logic::modes::bitcoin_vanity::BitcoinVanityKeyRequest {
                prefix: &first,
                suffix: &second,
                thread_idx: index,
                rng_seed: seed,
            },
        );
        let mut encoded = vec![0xa5; 64];
        encoded[..r.encoded_len].copy_from_slice(&r.encoded_public_key[..r.encoded_len]);
        Expected {
            matched: r.matches,
            payloads: vec![
                r.private_key.to_vec(),
                r.public_key.to_vec(),
                r.public_key_hash.to_vec(),
                encoded,
                (r.encoded_len as u32).to_le_bytes().to_vec(),
            ],
        }
    })
}

pub const ENTRY: &str = "kernel_find_bitcoin_vanity_private_key";
pub const PAYLOAD_SIZES: &[usize] = &[32, 33, 20, 64, 4];
