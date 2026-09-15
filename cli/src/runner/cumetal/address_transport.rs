use super::{CumetalRunner, Error, driver::Driver};
use crate::{args::Command, common::GlobalStats};
use std::{rc::Rc, sync::Arc};

pub(crate) struct Expected {
    pub(crate) matched: bool,
    pub(crate) payloads: Vec<Vec<u8>>,
}
fn entry(command: &Command) -> &'static str {
    match command {
        #[cfg(feature = "shallenge")]
        Command::Shallenge { .. } => crate::modes::shallenge::cumetal::ENTRY,
        #[cfg(feature = "solana")]
        Command::SolanaVanity { .. } => crate::modes::solana::cumetal::ENTRY,
        #[cfg(feature = "ethereum")]
        Command::EthereumVanity { .. } => crate::modes::ethereum::cumetal::ENTRY,
        #[cfg(feature = "bitcoin")]
        Command::BitcoinVanity { .. } => crate::modes::bitcoin::cumetal::ENTRY,
        #[cfg(feature = "self_test_support")]
        Command::SelfTest => unreachable!(),
        #[cfg(feature = "crypto-cli")]
        _ => unreachable!("cryptographic commands use their own batch transport"),
    }
}
fn payload_sizes(command: &Command) -> &'static [usize] {
    match command {
        #[cfg(feature = "shallenge")]
        Command::Shallenge { .. } => crate::modes::shallenge::cumetal::PAYLOAD_SIZES,
        #[cfg(feature = "solana")]
        Command::SolanaVanity { .. } => crate::modes::solana::cumetal::PAYLOAD_SIZES,
        #[cfg(feature = "ethereum")]
        Command::EthereumVanity { .. } => crate::modes::ethereum::cumetal::PAYLOAD_SIZES,
        #[cfg(feature = "bitcoin")]
        Command::BitcoinVanity { .. } => crate::modes::bitcoin::cumetal::PAYLOAD_SIZES,
        #[cfg(feature = "self_test_support")]
        Command::SelfTest => unreachable!(),
        #[cfg(feature = "crypto-cli")]
        _ => unreachable!("cryptographic commands use their own batch transport"),
    }
}
fn print_payloads(command: &Command, output: &[Vec<u8>]) -> Result<(), Error> {
    match command {
        #[cfg(feature = "shallenge")]
        Command::Shallenge { .. } => crate::modes::shallenge::cumetal::print_payloads(output)?,
        #[cfg(feature = "solana")]
        Command::SolanaVanity { .. } => crate::modes::solana::cumetal::print_payloads(output)?,
        #[cfg(feature = "ethereum")]
        Command::EthereumVanity { .. } => crate::modes::ethereum::cumetal::print_payloads(output)?,
        #[cfg(feature = "bitcoin")]
        Command::BitcoinVanity { .. } => crate::modes::bitcoin::cumetal::print_payloads(output)?,
        #[cfg(feature = "self_test_support")]
        Command::SelfTest => unreachable!(),
        #[cfg(feature = "crypto-cli")]
        _ => unreachable!("cryptographic commands use their own batch transport"),
    }
    Ok(())
}

fn inputs(command: &Command) -> Result<(Vec<u8>, Vec<u8>), Error> {
    Ok(match command {
        #[cfg(feature = "shallenge")]
        Command::Shallenge {
            username,
            target_hash,
        } => crate::modes::shallenge::cumetal::inputs(username, target_hash)?,
        #[cfg(feature = "solana")]
        Command::SolanaVanity { prefix, suffix } => {
            crate::modes::solana::cumetal::inputs(prefix, suffix)?
        }
        #[cfg(feature = "ethereum")]
        Command::EthereumVanity { prefix, suffix } => {
            crate::modes::ethereum::cumetal::inputs(prefix, suffix)?
        }
        #[cfg(feature = "bitcoin")]
        Command::BitcoinVanity { prefix, suffix } => {
            crate::modes::bitcoin::cumetal::inputs(prefix, suffix)?
        }
        #[cfg(feature = "self_test_support")]
        Command::SelfTest => unreachable!(),
        #[cfg(feature = "crypto-cli")]
        _ => unreachable!("cryptographic commands use their own batch transport"),
    })
}

#[cfg(all(test, feature = "ethereum"))]
mod input_tests {
    use super::*;

    #[test]
    fn ethereum_hex_patterns_match_known_seed_address() {
        let command = Command::EthereumVanity {
            prefix: "5395".into(),
            suffix: "279A".into(),
        };
        let (prefix, suffix) = inputs(&command).unwrap();
        let candidate = logic::modes::ethereum_vanity::generate_and_check_ethereum_vanity_key(
            &logic::modes::ethereum_vanity::EthereumVanityKeyRequest {
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
            assert!(
                inputs(&Command::EthereumVanity {
                    prefix: invalid.into(),
                    suffix: String::new(),
                })
                .is_err()
            );
        }
    }
}

fn expected(command: &Command, seed: u64, index: usize) -> Result<Expected, Error> {
    let (first, second) = inputs(command)?;
    Ok(match command {
        #[cfg(feature = "shallenge")]
        Command::Shallenge { .. } => {
            crate::modes::shallenge::cumetal::expected(&first, &second, seed, index)?
        }
        #[cfg(feature = "solana")]
        Command::SolanaVanity { .. } => {
            crate::modes::solana::cumetal::expected(&first, &second, seed, index)?
        }
        #[cfg(feature = "ethereum")]
        Command::EthereumVanity { .. } => {
            crate::modes::ethereum::cumetal::expected(&first, &second, seed, index)?
        }
        #[cfg(feature = "bitcoin")]
        Command::BitcoinVanity { .. } => {
            crate::modes::bitcoin::cumetal::expected(&first, &second, seed, index)?
        }
        #[cfg(feature = "self_test_support")]
        Command::SelfTest => unreachable!(),
        #[cfg(feature = "crypto-cli")]
        _ => unreachable!("cryptographic commands use their own batch transport"),
    })
}

impl CumetalRunner {
    pub(super) fn address_search(
        &self,
        command: &Command,
        driver: &Rc<Driver>,
        stats: Arc<GlobalStats>,
    ) -> Result<(), Error> {
        let module = self.module(&driver, entry(command))?;
        let candidates = self.options.blocks * self.options.threads_per_block;
        #[allow(unused_mut)]
        let mut command = command.clone();
        let mut batch = 0u64;
        loop {
            let seed = self
                .options
                .seed
                .map(|s| s.wrapping_add(batch))
                .unwrap_or_else(rand::random);
            let (first, second) = inputs(&command)?;
            let first_buffer = driver.buffer(&first)?;
            let second_buffer = driver.buffer(&second)?;
            let count = driver.buffer(&0u32.to_le_bytes())?;
            let payloads: Vec<_> = payload_sizes(&command)
                .iter()
                .map(|n| driver.buffer(&vec![0xa5; *n]))
                .collect::<Result<_, _>>()?;
            let thread = driver.buffer(&[0xa5; 4])?;
            #[allow(unused_mut)]
            let mut args = vec![
                first_buffer.pointer(),
                first.len() as u64,
                second_buffer.pointer(),
                second.len() as u64,
                seed,
            ];
            #[cfg(feature = "shallenge")]
            if matches!(command, Command::Shallenge { .. }) {
                args = vec![
                    first_buffer.pointer(),
                    first.len() as u64,
                    second_buffer.pointer(),
                    seed,
                ];
            }
            args.push(count.pointer());
            args.extend(payloads.iter().map(|b| b.pointer()));
            args.push(thread.pointer());
            module.launch(
                &mut args,
                self.options.blocks,
                self.options.threads_per_block,
            )?;
            if first_buffer.read()? != first || second_buffer.read()? != second {
                return Err("kernel changed an input buffer".into());
            }
            let matches = u32::from_le_bytes(count.read()?.try_into().unwrap());
            if matches > candidates {
                return Err("match count exceeds candidate count".into());
            }
            let output: Vec<_> = payloads
                .iter()
                .map(|b| b.read())
                .collect::<Result<_, _>>()?;
            let thread_bytes = thread.read()?;
            if self.options.verify {
                let mut expected_count = 0;
                for i in 0..candidates {
                    if expected(&command, seed, i as usize)?.matched {
                        expected_count += 1;
                    }
                }
                if matches != expected_count {
                    return Err(format!("match count: GPU {matches}, CPU {expected_count}").into());
                }
            }
            if matches == 0 {
                if output
                    .iter()
                    .flatten()
                    .chain(thread_bytes.iter())
                    .any(|b| *b != 0xa5)
                {
                    return Err("no-match launch modified a result payload".into());
                }
            } else {
                let selected = u32::from_le_bytes(thread_bytes.try_into().unwrap());
                if selected >= candidates {
                    return Err("returned thread index outside launch".into());
                }
                // Always validate the returned key/nonce, even without a full CPU sweep.
                let reference = expected(&command, seed, selected as usize)?;
                if !reference.matched {
                    return Err(format!("GPU reported a nonmatching CPU candidate for thread {selected}, seed {seed}").into());
                }
                if output.len() != reference.payloads.len() {
                    return Err("GPU output payload count differs from CPU reference".into());
                }
                let differences: Vec<_> = output
                    .iter()
                    .zip(&reference.payloads)
                    .enumerate()
                    .filter(|(_, (actual, expected))| actual != expected)
                    .map(|(index, (actual, expected))| {
                        let offset = actual.iter().zip(expected).position(|(a, b)| a != b);
                        format!(
                            "payload {index}: first differing byte {offset:?}, lengths {}/{}",
                            actual.len(),
                            expected.len()
                        )
                    })
                    .collect();
                if !differences.is_empty() {
                    return Err(format!(
                        "GPU result differs from CPU for thread {selected}, seed {seed}: {}",
                        differences.join("; ")
                    )
                    .into());
                }
                println!("MATCH thread={selected} seed={seed}");
                print_payloads(&command, &output)?;
                #[cfg(feature = "shallenge")]
                if let Command::Shallenge { target_hash, .. } = &mut command {
                    *target_hash = hex::encode(&output[0]);
                }
            }
            stats.add_launch(candidates as usize);
            stats.add_matches(matches as usize);
            println!(
                "CUMETAL_BATCH batch={batch} seed={seed} candidates={candidates} matches={matches} verified={} guards=intact",
                self.options.verify
            );
            batch += 1;
            if self.options.batches.is_some_and(|limit| batch >= limit) {
                break;
            }
        }
        Ok(())
    }
}
