use crate::runner::cumetal::{Error, address_transport::Expected};

pub fn print_payloads(output: &[Vec<u8>]) -> Result<(), Error> {
    {
        let length = u64::from_le_bytes(output[2].as_slice().try_into()?);
        let nonce = output[1]
            .get(..usize::try_from(length)?)
            .ok_or("invalid nonce length")?;
        println!("hash={}", hex::encode(&output[0]));
        println!("nonce={}", std::str::from_utf8(nonce)?);
    };
    Ok(())
}

pub fn inputs(username: &str, target_hash: &str) -> Result<(Vec<u8>, Vec<u8>), Error> {
    Ok((username.as_bytes().to_vec(), hex::decode(target_hash)?))
}

pub fn expected(first: &[u8], second: &[u8], seed: u64, index: usize) -> Result<Expected, Error> {
    Ok({
        let target: [u8; 32] = second
            .try_into()
            .map_err(|_| "target must contain 32 bytes")?;
        let r = logic::modes::shallenge::generate_and_check_shallenge(
            &logic::modes::shallenge::ShallengeRequest {
                username: &first,
                username_len: first.len(),
                target_hash: &target,
                thread_idx: index,
                rng_seed: seed,
            },
        );
        Expected {
            matched: r.is_better,
            payloads: vec![
                r.hash.to_vec(),
                r.nonce.to_vec(),
                (r.nonce_len as u64).to_le_bytes().to_vec(),
            ],
        }
    })
}

pub const ENTRY: &str = "kernel_find_better_shallenge_nonce";
pub const PAYLOAD_SIZES: &[usize] = &[32, 64, 8];
