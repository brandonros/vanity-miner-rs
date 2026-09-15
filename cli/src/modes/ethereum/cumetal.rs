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
        let r = logic::modes::ethereum::generate_and_check_ethereum_vanity_key(
            &logic::modes::ethereum::EthereumVanityKeyRequest {
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

pub fn run(
    runner: &crate::runner::cumetal::CumetalRunner,
    prefix: &str,
    suffix: &str,
    driver: &std::rc::Rc<crate::runner::cumetal::driver::Driver>,
    stats: std::sync::Arc<crate::runner::progress::GlobalStats>,
) -> Result<(), Error> {
    use crate::runner::cumetal::address_transport::{AddressBatch, ParameterLayout};
    let (first, second) = inputs(prefix, suffix)?;
    runner.address_search(
        AddressBatch {
            entry: ENTRY,
            payload_sizes: PAYLOAD_SIZES,
            first,
            second,
            layout: ParameterLayout::Patterns,
            reference: expected,
            print: print_payloads,
        },
        driver,
        stats,
    )
}

#[cfg(test)]
mod input_tests {
    use super::*;

    #[test]
    fn ethereum_hex_patterns_match_known_seed_address() {
        let (prefix, suffix) = inputs("5395", "279A").unwrap();
        let candidate = logic::modes::ethereum::generate_and_check_ethereum_vanity_key(
            &logic::modes::ethereum::EthereumVanityKeyRequest {
                prefix: &prefix,
                suffix: &suffix,
                thread_idx: 0,
                rng_seed: 1,
            },
        );
        assert_eq!(
            hex::encode(candidate.address),
            "539571f1569bfcb63397630dd2e7765555ae279a"
        );
        assert!(candidate.matches, "known nonempty hex patterns must match");
        for invalid in ["539", "zz"] {
            assert!(inputs(invalid, "").is_err());
        }
    }
}
