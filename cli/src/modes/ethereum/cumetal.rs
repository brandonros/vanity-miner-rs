use crate::runner::cumetal::{Error, address_transport::Expected};

pub fn print_payloads(output: &[Vec<u8>]) -> Result<(), Error> {
    {
        println!("private_key={}", hex::encode(&output[0]));
        println!("public_key={}", hex::encode(&output[1]));
        println!("address=0x{}", hex::encode(&output[2]));
    };
    Ok(())
}

pub fn inputs(prefix: &str, suffix: &str) -> Result<(Vec<u8>, Vec<u8>), Error> {
    Ok((hex::decode(prefix)?, hex::decode(suffix)?))
}

pub fn expected(first: &[u8], second: &[u8], seed: u64, index: usize) -> Result<Expected, Error> {
    Ok({
        let r = logic::modes::ethereum_vanity::generate_and_check_ethereum_vanity_key(
            &logic::modes::ethereum_vanity::EthereumVanityKeyRequest {
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
                r.address.to_vec(),
            ],
        }
    })
}

pub const ENTRY: &str = "kernel_find_ethereum_vanity_private_key";
pub const PAYLOAD_SIZES: &[usize] = &[32, 64, 20];
