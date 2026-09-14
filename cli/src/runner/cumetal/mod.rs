//! CuMetal host backend: consume prebuilt PTX without a local NVIDIA toolchain.
mod cache;
mod driver;
use crate::{args::Command, common::GlobalStats, runner::Runner};
use clap::Args;
use driver::{Driver, Module};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};
pub type Error = Box<dyn std::error::Error + Send + Sync>;

#[derive(Args, Clone)]
pub struct CumetalOptions {
    /// CuMetal driver library (libcumetal.dylib).
    #[arg(long, global = true, default_value = "libcumetal.dylib")]
    pub cumetal_library: PathBuf,
    /// Original Rust-CUDA PTX artifact to compile for the selected entry.
    #[arg(long, global = true)]
    pub ptx: Option<PathBuf>,
    /// Directory of precompiled ENTRY.metal files and their ABI sidecars.
    #[arg(long, global = true)]
    pub module_dir: Option<PathBuf>,
    #[arg(long, global = true, default_value = "cumetalc")]
    pub cumetalc: PathBuf,
    #[arg(long, global = true, default_value = ".cumetal-cache")]
    pub cumetal_cache: PathBuf,
    /// Stop after this many launches; omitted means keep searching.
    #[arg(long, global=true, value_parser=clap::value_parser!(u64).range(1..))]
    pub batches: Option<u64>,
    #[arg(long, global=true, default_value_t=32, value_parser=clap::value_parser!(u32).range(1..=1024))]
    pub threads_per_block: u32,
    #[arg(long, global=true, default_value_t=1, value_parser=clap::value_parser!(u32).range(1..=65535))]
    pub blocks: u32,
    /// Starting deterministic seed; subsequent batches increment it.
    #[arg(long, global = true)]
    pub seed: Option<u64>,
    /// Compare every candidate and the match count with the CPU reference.
    #[arg(long, global = true)]
    pub verify: bool,
    /// Run only selected self-test slots (repeatable); omitted runs all slots.
    #[arg(long, global=true, value_parser=clap::value_parser!(u32).range(0..118))]
    pub self_test_slot: Vec<u32>,
}

pub struct CumetalRunner {
    options: CumetalOptions,
}
impl CumetalRunner {
    pub fn new(options: CumetalOptions) -> Result<Self, Error> {
        if options.ptx.is_none() && options.module_dir.is_none() {
            return Err("Specify --ptx or --module-dir for CuMetal".into());
        }
        options
            .blocks
            .checked_mul(options.threads_per_block)
            .ok_or("launch size overflow")?;
        Ok(Self { options })
    }
    fn module(&self, driver: &Rc<Driver>, entry: &str) -> Result<Module, Error> {
        let path = if let Some(directory) = &self.options.module_dir {
            directory.join(format!("{entry}.metal"))
        } else {
            let input = self.options.ptx.as_ref().ok_or("--ptx is required")?;
            let compiler = resolve_program(&self.options.cumetalc)?;
            let mut hash = Sha256::new();
            let ptx_bytes = std::fs::read(input)?;
            let compiler_bytes = std::fs::read(&compiler)?;
            hash.update(&ptx_bytes);
            hash.update(&compiler_bytes);
            hash.update(b"cumetal-ir/ptx-strict/msl/v2");
            hash.update(entry.as_bytes());
            let directory = self
                .options
                .cumetal_cache
                .join(hex::encode(hash.finalize()));
            cache::module(&directory, entry, |output| {
                eprintln!("Compiling {entry} from {}", input.display());
                // Compile the exact bytes used in the cache key even if the
                // supplied artifact is replaced while the compiler is running.
                let snapshot = output.with_extension("ptx");
                std::fs::write(&snapshot, &ptx_bytes)?;
                let status = std::process::Command::new(&compiler)
                    .arg(&snapshot)
                    .args([
                        "--backend=cumetal-ir",
                        "--ptx-strict",
                        "--overwrite",
                        "--entry",
                        entry,
                        "--emit=msl",
                        "-o",
                    ])
                    .arg(output)
                    .status()?;
                if !status.success() {
                    return Err(format!("CuMetal compilation failed for {entry}: {status}").into());
                }
                if std::fs::read(&compiler)? != compiler_bytes {
                    return Err(
                        "CuMetal compiler changed during compilation; retry with a stable compiler"
                            .into(),
                    );
                }
                std::fs::remove_file(snapshot)?;
                Ok(())
            })?
        };
        if !path.with_extension("metal.cumetal-abi").is_file() {
            return Err(format!("Missing ABI sidecar for {}", path.display()).into());
        }
        driver.module(&path, entry)
    }
}
fn resolve_program(path: &Path) -> Result<PathBuf, Error> {
    if path.components().count() > 1 || path.is_file() {
        return Ok(path.canonicalize()?);
    }
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let candidate = directory.join(path);
        if candidate.is_file() {
            return Ok(candidate.canonicalize()?);
        }
    }
    Err(format!("Cannot find {} on PATH", path.display()).into())
}
struct Expected {
    matched: bool,
    payloads: Vec<Vec<u8>>,
}
fn entry(command: &Command) -> &'static str {
    match command {
        #[cfg(feature = "shallenge")]
        Command::Shallenge { .. } => "kernel_find_better_shallenge_nonce",
        #[cfg(feature = "solana")]
        Command::SolanaVanity { .. } => "kernel_find_solana_vanity_private_key",
        #[cfg(feature = "ethereum")]
        Command::EthereumVanity { .. } => "kernel_find_ethereum_vanity_private_key",
        #[cfg(feature = "bitcoin")]
        Command::BitcoinVanity { .. } => "kernel_find_bitcoin_vanity_private_key",
        #[cfg(feature = "self_test")]
        Command::SelfTest => unreachable!(),
    }
}
fn payload_sizes(command: &Command) -> &'static [usize] {
    match command {
        #[cfg(feature = "shallenge")]
        Command::Shallenge { .. } => &[32, 64, 8],
        #[cfg(feature = "solana")]
        Command::SolanaVanity { .. } => &[32, 32, 64],
        #[cfg(feature = "ethereum")]
        Command::EthereumVanity { .. } => &[32, 64, 20],
        #[cfg(feature = "bitcoin")]
        Command::BitcoinVanity { .. } => &[32, 33, 20, 64, 4],
        #[cfg(feature = "self_test")]
        Command::SelfTest => unreachable!(),
    }
}
fn print_payloads(command: &Command, output: &[Vec<u8>]) -> Result<(), Error> {
    match command {
        #[cfg(feature = "shallenge")]
        Command::Shallenge { .. } => {
            let length = u64::from_le_bytes(output[2].as_slice().try_into()?);
            let nonce = output[1]
                .get(..usize::try_from(length)?)
                .ok_or("invalid nonce length")?;
            println!("hash={}", hex::encode(&output[0]));
            println!("nonce={}", std::str::from_utf8(nonce)?);
        }
        #[cfg(feature = "solana")]
        Command::SolanaVanity { .. } => {
            let length = output[2]
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(output[2].len());
            println!("private_key={}", hex::encode(&output[0]));
            println!("public_key={}", hex::encode(&output[1]));
            println!("address={}", std::str::from_utf8(&output[2][..length])?);
        }
        #[cfg(feature = "ethereum")]
        Command::EthereumVanity { .. } => {
            println!("private_key={}", hex::encode(&output[0]));
            println!("public_key={}", hex::encode(&output[1]));
            println!("address=0x{}", hex::encode(&output[2]));
        }
        #[cfg(feature = "bitcoin")]
        Command::BitcoinVanity { .. } => {
            let length = u32::from_le_bytes(output[4].as_slice().try_into()?) as usize;
            let address = output[3].get(..length).ok_or("invalid address length")?;
            println!("private_key={}", hex::encode(&output[0]));
            println!("public_key={}", hex::encode(&output[1]));
            println!("hash160={}", hex::encode(&output[2]));
            println!("address={}", std::str::from_utf8(address)?);
        }
        #[cfg(feature = "self_test")]
        Command::SelfTest => unreachable!(),
    }
    Ok(())
}

fn inputs(command: &Command) -> Result<(Vec<u8>, Vec<u8>), Error> {
    Ok(match command {
        #[cfg(feature = "shallenge")]
        Command::Shallenge {
            username,
            target_hash,
        } => (username.as_bytes().to_vec(), hex::decode(target_hash)?),
        #[cfg(feature = "solana")]
        Command::SolanaVanity { prefix, suffix } => {
            (prefix.as_bytes().to_vec(), suffix.as_bytes().to_vec())
        }
        #[cfg(feature = "ethereum")]
        Command::EthereumVanity { prefix, suffix } => (hex::decode(prefix)?, hex::decode(suffix)?),
        #[cfg(feature = "bitcoin")]
        Command::BitcoinVanity { prefix, suffix } => {
            (prefix.as_bytes().to_vec(), suffix.as_bytes().to_vec())
        }
        #[cfg(feature = "self_test")]
        Command::SelfTest => unreachable!(),
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
        let candidate =
            logic::generate_and_check_ethereum_vanity_key(&logic::EthereumVanityKeyRequest {
                prefix: &prefix,
                suffix: &suffix,
                thread_idx: 0,
                rng_seed: 1,
            });
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
            let target: [u8; 32] = second
                .try_into()
                .map_err(|_| "target must contain 32 bytes")?;
            let r = logic::generate_and_check_shallenge(&logic::ShallengeRequest {
                username: &first,
                username_len: first.len(),
                target_hash: &target,
                thread_idx: index,
                rng_seed: seed,
            });
            Expected {
                matched: r.is_better,
                payloads: vec![
                    r.hash.to_vec(),
                    r.nonce.to_vec(),
                    (r.nonce_len as u64).to_le_bytes().to_vec(),
                ],
            }
        }
        #[cfg(feature = "solana")]
        Command::SolanaVanity { .. } => {
            let r = logic::generate_and_check_solana_vanity_key(&logic::SolanaVanityKeyRequest {
                prefix: &first,
                suffix: &second,
                thread_idx: index,
                rng_seed: seed,
            });
            Expected {
                matched: r.matches,
                payloads: vec![
                    r.private_key.to_vec(),
                    r.public_key.to_vec(),
                    r.encoded_public_key.to_vec(),
                ],
            }
        }
        #[cfg(feature = "ethereum")]
        Command::EthereumVanity { .. } => {
            let r =
                logic::generate_and_check_ethereum_vanity_key(&logic::EthereumVanityKeyRequest {
                    prefix: &first,
                    suffix: &second,
                    thread_idx: index,
                    rng_seed: seed,
                });
            Expected {
                matched: r.matches,
                payloads: vec![
                    r.private_key.to_vec(),
                    r.public_key.to_vec(),
                    r.address.to_vec(),
                ],
            }
        }
        #[cfg(feature = "bitcoin")]
        Command::BitcoinVanity { .. } => {
            let r = logic::generate_and_check_bitcoin_vanity_key(&logic::BitcoinVanityKeyRequest {
                prefix: &first,
                suffix: &second,
                thread_idx: index,
                rng_seed: seed,
            });
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
        }
        #[cfg(feature = "self_test")]
        Command::SelfTest => unreachable!(),
    })
}
impl Runner for CumetalRunner {
    fn device_count(&self) -> usize {
        1
    }
    fn run(&self, command: &Command, stats: Arc<GlobalStats>) -> Result<(), Error> {
        let driver = Driver::open(&self.options.cumetal_library)?;
        #[cfg(feature = "self_test")]
        if matches!(command, Command::SelfTest) {
            return self.self_tests(&driver);
        }
        let module = self.module(&driver, entry(command))?;
        let candidates = self.options.blocks * self.options.threads_per_block;
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
#[cfg(feature = "self_test")]
impl CumetalRunner {
    fn self_tests(&self, driver: &Rc<Driver>) -> Result<(), Error> {
        use vanity_miner::self_test_suite::{self, Kind, Outcome};
        self_test_suite::run("CuMetal", |case| {
            let slot = match case.kind {
                Kind::Probe => 0,
                Kind::Legacy(slot) => {
                    if !self.options.self_test_slot.is_empty() && !self.options.self_test_slot.contains(&(slot as u32)) {
                        return Ok(Outcome::Skipped("not selected"));
                    }
                    slot
                }
                _ => return Ok(Outcome::Skipped("crypto candidate transport is not supported by CuMetal")),
            };
            let name = case.kernel;
            let operation = (|| -> Result<(), Error> {
                let module = self.module(driver, name)?;
                let result = driver.buffer(&vec![0xa5; logic::SELF_TEST_NUM_CHECKS * 4])?;
                module.launch(&mut [result.pointer()], 1, 1)?;
                let bytes = result.read()?;
                for (index, word) in bytes.chunks_exact(4).enumerate() {
                    let value = u32::from_le_bytes(word.try_into().unwrap());
                    let expected = if index == slot { 1 } else { 0xa5a5a5a5 };
                    if value != expected { return Err(format!("{name}: slot {index}: got {value}, expected {expected}").into()); }
                }
                println!("NUMERICAL_PASS kernel={name} slot={slot}; other slots and guards intact");
                Ok(())
            })();
            operation.map_err(|e| e.to_string())?;
            Ok(Outcome::Passed)
        }).map_err(Into::into)
    }
}
