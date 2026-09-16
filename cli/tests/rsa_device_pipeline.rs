#![cfg(feature = "rsa-modulus")]
use logic::modes::rsa_modulus::{self as pipeline, SearchConfig, Task};
use num_bigint_dig::{BigUint, ModInverse};

fn bytes<const N: usize>(n: &BigUint) -> [u8; N] {
    let raw = n.to_bytes_be();
    let mut out = [0; N];
    out[N - raw.len()..].copy_from_slice(&raw);
    out
}

fn config(prefix: &str, suffix: &str) -> SearchConfig {
    let one = BigUint::from(1u8);
    let (lower, upper) = if prefix.is_empty() {
        (&one << 2047, (&one << 2048) - &one)
    } else {
        let prefix_value = BigUint::parse_bytes(prefix.as_bytes(), 16).unwrap();
        let shift = 2048 - 4 * prefix.len();
        (
            &prefix_value << shift,
            ((prefix_value + &one) << shift) - &one,
        )
    };
    let max = (&one << 1024) - &one;
    let mut min = ((&lower + &max - &one) / &max).max(&one << 1023);
    if &min % 2u8 == BigUint::from(0u8) {
        min += &one;
    }
    let p_count = if min <= max {
        (&max - &min) / 2u8 + &one
    } else {
        one.clone()
    };
    SearchConfig {
        lower: bytes(&lower),
        upper: bytes(&upper),
        p_min: bytes(&min),
        p_count: bytes(&p_count),
        suffix: bytes(&if suffix.is_empty() {
            one
        } else {
            BigUint::parse_bytes(suffix.as_bytes(), 16).unwrap()
        }),
        seed: [42; 32],
        worker: 7,
        suffix_bits: (suffix.len() * 4).max(1) as u32,
        reserved: 0,
    }
}

// Independent heap-backed reference; deliberately uses division/modular inverse
// rather than the device implementation's masking and fixed-width operations.
fn reference(c: &SearchConfig, p: &BigUint) -> Option<(BigUint, BigUint)> {
    let one = BigUint::from(1u8);
    let lower = BigUint::from_bytes_be(&c.lower);
    let upper = BigUint::from_bytes_be(&c.upper);
    let min = ((lower + p - &one) / p).max(&one << 1023);
    let max = (upper / p).min((&one << 1024) - &one);
    if min > max {
        return None;
    }
    let stride = &one << c.suffix_bits as usize;
    let inverse = p.mod_inverse(&stride)?.to_biguint()?;
    let residue = (BigUint::from_bytes_be(&c.suffix) * inverse) % &stride;
    let first = &min + ((residue + &stride - (&min % &stride)) % &stride);
    if first > max {
        return None;
    }
    let count = (max - &first) / stride + one;
    Some((first, count))
}

fn compare(c: &SearchConfig, p: &BigUint) {
    let actual = pipeline::progression(c, &bytes(p)).map(|(first, count)| {
        (
            BigUint::from_bytes_be(&first),
            BigUint::from_bytes_be(&count),
        )
    });
    assert_eq!(actual, reference(c, p));
}

#[test]
fn range_division_boundaries_match_biguint() {
    let one = BigUint::from(1u8);
    let zero = BigUint::from(0u8);
    let p = (&one << 1024) - BigUint::from(109u8);
    let q_min = &one << 1023;
    let q_max = (&one << 1024) - &one;
    for q in [&q_min - &one, q_min.clone(), &q_min + &one, q_max] {
        for remainder in [zero.clone(), one.clone(), &p - &one] {
            // Exercise both sides of r + width == p and width == p.
            for width in [
                zero.clone(),
                one.clone(),
                &p - &remainder - &one,
                &p - &remainder,
                &p - &one,
                p.clone(),
                &p + &one,
            ] {
                for suffix in ["", "1", "3"] {
                    let mut c = config("", suffix);
                    let lower = &p * &q + &remainder;
                    c.lower = bytes(&lower);
                    c.upper = bytes(&(&lower + &width));
                    compare(&c, &p);
                }
            }
        }
    }
    let mut inverted = config("", "");
    inverted.lower = inverted.upper;
    inverted.upper = [0; 256];
    assert!(pipeline::progression(&inverted, &bytes(&p)).is_none());
}

#[test]
#[ignore = "manual range-construction microbenchmark; run with --nocapture"]
fn benchmark_narrow_range_construction() {
    let c = config(&format!("80{}", "3132333435363738".repeat(16)), "");
    let candidates: Vec<_> = (0..4096)
        .map(|id| pipeline::generate_p(&c, id).unwrap())
        .collect();
    for _ in 0..3 {
        let start = std::time::Instant::now();
        let mut nonempty = 0usize;
        for _ in 0..16 {
            for p in &candidates {
                nonempty += std::hint::black_box(pipeline::progression(
                    std::hint::black_box(&c),
                    std::hint::black_box(p),
                ))
                .is_some() as usize;
            }
        }
        eprintln!(
            "65536 ranges in {:.6}s ({:.2}/sec), {nonempty} nonempty",
            start.elapsed().as_secs_f64(),
            65536.0 / start.elapsed().as_secs_f64()
        );
    }
}

#[test]
fn device_ranges_match_biguint_including_full_width_patterns() {
    let one = BigUint::from(1u8);
    let p = (&one << 1024) - BigUint::from(109u8);
    let q = (&one << 1023) + BigUint::from(123u8);
    let n = hex::encode(bytes::<256>(&(&p * &q)));
    for prefix_len in [0, 1, 3, 128, 255, 256, 257, 258, 384, 511, 512] {
        for suffix_len in [0, 1, 3, 127, 255, 256, 257, 384, 511, 512] {
            let c = config(&n[..prefix_len], &n[512 - suffix_len..]);
            compare(&c, &p);
            // Most different p's have empty ranges for a tightly fixed modulus.
            for id in 0..4 {
                let other = pipeline::generate_p(&c, id).unwrap();
                compare(&c, &BigUint::from_bytes_be(&other));
            }
        }
    }
    // Ceiling divisions close to 2^2048 must not add p in the narrow type.
    for prefix in [
        "f",
        "ffffffffffffffffffffffff",
        "fffffffffffffffffffffffffe",
    ] {
        let c = config(prefix, "1");
        compare(&c, &p);
        compare(&c, &q);
    }
}

#[test]
fn narrow_patterns_check_ranges_before_primality() {
    assert!(!pipeline::range_first(&config("abc", "1")));
    assert!(pipeline::range_first(&config(
        &format!("a{}", "0".repeat(255)),
        "1"
    )));
    assert!(pipeline::range_first(&config(
        &format!("a{}", "0".repeat(127)),
        &format!("{}1", "0".repeat(127))
    )));
    assert!(pipeline::range_first(&config(
        "",
        &format!("{}1", "0".repeat(511))
    )));
}

#[test]
fn device_generation_is_bounded_and_domain_separated() {
    let mut c = config("ffffffffffffffffffffffff", "");
    let min = BigUint::from_bytes_be(&c.p_min);
    let mut seen = std::collections::HashSet::new();
    for worker in 0..3 {
        c.worker = worker;
        for id in 0..100 {
            let p = pipeline::generate_p(&c, id).unwrap();
            assert_eq!(pipeline::generate_p(&c, id), Some(p));
            assert_eq!(p[127] & 1, 1);
            assert_eq!(p[0] & 128, 128);
            assert!(BigUint::from_bytes_be(&p) >= min);
            assert!(seen.insert(p));
        }
    }
}

#[test]
fn tiles_cover_short_ranges_once_and_retire_winners() {
    let c = config("", "");
    let first = (BigUint::from(1u8) << 1023) + BigUint::from(1u8);
    for count in [1u32, 2, 3, 63, 64, 65, 257, 1000] {
        for tile in [1u32, 7, 64, 256] {
            let mut task = Task {
                state: 2,
                first: bytes(&first),
                count: bytes(&BigUint::from(count)),
                remaining: bytes(&BigUint::from(count)),
                cursor: bytes(&BigUint::from(count - 1)),
                ..Task::EMPTY
            };
            let mut seen = std::collections::BTreeSet::new();
            while task.state != 0 {
                for offset in 0..tile {
                    if let Some(q) = pipeline::q_at(&c, &task, offset) {
                        assert!(seen.insert(BigUint::from_bytes_be(&q)));
                    }
                }
                pipeline::finish_tile(&mut task, tile);
            }
            assert_eq!(seen.len(), count as usize);
            assert_eq!(seen.first(), Some(&first));
            assert_eq!(
                seen.last(),
                Some(&(&first + BigUint::from(count - 1) * 2u8))
            );
        }
    }
    let mut task = Task {
        state: 2,
        winner: 1,
        p: [42; 128],
        ..Task::EMPTY
    };
    pipeline::finish_tile(&mut task, 1);
    assert_eq!(task.state, 0);
    assert_eq!(task.p, [0; 128]);
}

#[test]
fn range_cursors_do_not_truncate_to_machine_integers() {
    let count = BigUint::from(1u8) << 1000;
    let mut task = Task {
        state: 2,
        count: bytes(&count),
        remaining: bytes(&count),
        cursor: bytes(&(&count - BigUint::from(2u8))),
        ..Task::EMPTY
    };
    pipeline::finish_tile(&mut task, 7);
    assert_eq!(BigUint::from_bytes_be(&task.cursor), BigUint::from(5u8));
    assert_eq!(
        BigUint::from_bytes_be(&task.remaining),
        count - BigUint::from(7u8)
    );
}

#[test]
fn preparation_excludes_the_entire_forbidden_factor_interval() {
    let one = BigUint::from(1u8);
    let distance = &one << 924;
    let p = (&one << 1023) + (&one << 1022) + &one;
    let c = config("", "");
    let mut task = Task {
        state: 1,
        p: bytes(&p),
        id: 42,
        ..Task::EMPTY
    };
    assert!(pipeline::prepare_range(&c, &mut task).unwrap());
    assert!(BigUint::from_bytes_be(&task.skip_count) > BigUint::from(0u8));
    // Check both ends of both retained intervals by selecting compressed indices.
    let count = BigUint::from_bytes_be(&task.count);
    let skip = BigUint::from_bytes_be(&task.skip_start);
    for cursor in [BigUint::from(0u8), &skip - &one, skip, &count - &one] {
        task.cursor = bytes(&cursor);
        let q = BigUint::from_bytes_be(&pipeline::q_at(&c, &task, 0).unwrap());
        let delta = if p > q { &p - &q } else { &q - &p };
        assert!(delta > distance);
    }
    // A tightly constrained modulus whose only q equals p must be retired,
    // rather than remaining active forever while every candidate is rejected.
    let n = hex::encode(bytes::<256>(&(&p * &p)));
    for (prefix, suffix) in [(&n[..], ""), ("", &n[..])] {
        let c = config(prefix, suffix);
        let mut task = Task {
            state: 1,
            p: bytes(&p),
            ..Task::EMPTY
        };
        assert!(!pipeline::prepare_range(&c, &mut task).unwrap());
        assert_eq!(task.state, 0);
        assert_eq!(task.p, [0; 128]);
    }
}

#[test]
fn pipeline_host_reserves_unique_tasks_and_rejects_invalid_counts() {
    use vanity_miner::{
        modes::rsa_modulus::{ModulusSearch, pipeline as host},
        runner::session::SearchControl,
    };
    let search = ModulusSearch {
        prefix: "abc".into(),
        suffix: "1".into(),
        workers: 1,
    };
    let control = SearchControl::new();
    control.set_batch_size(17).unwrap();
    control.set_device_launch_limit(Some(3));
    let mut launches = Vec::new();
    let mut seed = None;
    host::run(
        &search,
        &control,
        &host::StageStats::default(),
        4,
        |config, _, start, capacity| {
            if let Some(previous) = seed {
                assert_eq!(config.seed, previous);
            }
            seed = Some(config.seed);
            launches.push(start);
            assert_eq!(capacity, 17);
            Ok((
                pipeline::Counts {
                    p_tested: capacity,
                    ..Default::default()
                },
                zeroize::Zeroizing::new(Vec::new()),
            ))
        },
    )
    .unwrap();
    assert_eq!(launches, [0, 68, 136]);
    assert_eq!(control.statistics().0, 51);

    let control = SearchControl::new();
    let result = host::run(
        &search,
        &control,
        &host::StageStats::default(),
        4,
        |_, _, _, capacity| {
            Ok((
                pipeline::Counts {
                    q_tested: capacity * 4 + 1,
                    ..Default::default()
                },
                zeroize::Zeroizing::new(Vec::new()),
            ))
        },
    );
    assert!(result.is_err());
    assert!(control.stopped());
}

#[test]
fn completed_pairs_are_independently_verified_before_export() {
    use rsa::{
        RsaPrivateKey,
        pkcs8::DecodePrivateKey,
        traits::{PrivateKeyParts, PublicKeyParts},
    };
    use vanity_miner::modes::rsa_modulus::pipeline::verify_pair;
    let key = RsaPrivateKey::new(&mut rand::rngs::OsRng, 2048).unwrap();
    let n = hex::encode(bytes::<256>(key.n()));
    let mut c = config(&n, "");
    // Public test fixture configuration: a single known p lets the test exercise
    // full verification without waiting for a stochastic search to finish.
    c.p_min = bytes(&key.primes()[0]);
    c.p_count = bytes(&BigUint::from(1u8));
    let pair = pipeline::Pair {
        p: c.p_min,
        q: bytes(&key.primes()[1]),
        id: 9,
    };
    let pattern = logic::search::hex_pattern::HexPattern::new(&n, "", 256).unwrap();
    // One step prepares p; the next launch must resume it without generating p again.
    let mut task = Task::EMPTY;
    let (prepared, found) = pipeline::mine(&c, &pattern, &mut task, 9, 1, 1);
    assert!(found.is_none());
    assert_eq!(
        (
            prepared.p_tested,
            prepared.p_accepted,
            prepared.ranges,
            prepared.q_tested
        ),
        (1, 1, 1, 0)
    );
    assert_eq!(task.id, 9);
    let (searched, found) = pipeline::mine(&c, &pattern, &mut task, 10, 1, 64);
    assert_eq!(
        (searched.p_tested, searched.q_tested, searched.matches),
        (0, 1, 1)
    );
    assert!(found == Some(pair));
    assert!(task == Task::EMPTY);
    // Even a large budget stops after one match, bounding the output per lane.
    let (combined, found) = pipeline::mine(&c, &pattern, &mut task, 11, 1, 1024);
    assert_eq!(
        (combined.p_tested, combined.q_tested, combined.matches),
        (1, 1, 1)
    );
    assert_eq!(found.unwrap().id, 11);
    assert!(task == Task::EMPTY);
    let output = zeroize::Zeroizing::new(verify_pair(&c, &pattern, &pair).unwrap());
    let encoded = output
        .lines()
        .find_map(|line| line.strip_prefix("[rsa-modulus] private_key_pkcs8="))
        .unwrap();
    let der = zeroize::Zeroizing::new(hex::decode(encoded).unwrap());
    let recovered = RsaPrivateKey::from_pkcs8_der(&der).unwrap();
    assert_eq!(recovered.n(), key.n());
    recovered.validate().unwrap();

    let mut corrupt = pair;
    corrupt.p[1] ^= 1;
    assert!(verify_pair(&c, &pattern, &corrupt).is_err());
    let mut corrupt = pair;
    corrupt.q[127] &= 0xfe;
    assert!(verify_pair(&c, &pattern, &corrupt).is_err());
}

#[test]
fn mining_launches_resume_without_repeating_or_skipping_q_values() {
    let c = config("", "");
    let pattern = logic::search::hex_pattern::HexPattern::new("", "", 256).unwrap();
    let first = (BigUint::from(1u8) << 1023) + BigUint::from(1u8);
    let mut task = Task {
        // An undersized p ensures every q misses before primality testing.
        state: 2,
        id: 7,
        first: bytes(&first),
        count: bytes(&BigUint::from(9u8)),
        remaining: bytes(&BigUint::from(9u8)),
        cursor: bytes(&BigUint::from(7u8)),
        ..Task::EMPTY
    };
    let mut seen = std::collections::BTreeSet::new();
    let mut used = 0u32;
    for steps in [3, 2, 4] {
        // Independent expected progression across the wrap and launch boundaries.
        for offset in 0..steps {
            let expected = &first + BigUint::from((7 + used + offset) % 9) * 2u8;
            let actual = pipeline::q_at(&c, &task, offset).unwrap();
            assert_eq!(BigUint::from_bytes_be(&actual), expected);
            assert!(seen.insert(actual));
        }
        let (counts, found) =
            pipeline::mine(&c, &pattern, &mut task, 100 + u64::from(used), 1, steps);
        assert!(found.is_none());
        assert_eq!(
            (
                counts.p_tested,
                counts.q_tested,
                counts.active,
                counts.errors
            ),
            (0, steps, 1, 0)
        );
        used += steps;
        if used < 9 {
            assert_eq!(task.id, 7);
            assert_eq!(
                BigUint::from_bytes_be(&task.cursor),
                BigUint::from((7 + used) % 9)
            );
            assert_eq!(
                BigUint::from_bytes_be(&task.remaining),
                BigUint::from(9 - used)
            );
        }
    }
    assert_eq!(seen.len(), 9);
    assert!(task == Task::EMPTY);

    let mut refill_config = c;
    // Composite p divisible by 3: refills are guaranteed to miss deterministically.
    refill_config.p_min = bytes(&first);
    refill_config.p_count = bytes(&BigUint::from(1u8));
    let (counts, found) = pipeline::mine(&refill_config, &pattern, &mut task, 500, 17, 4);
    assert_eq!(
        (
            counts.p_tested,
            counts.p_accepted,
            counts.q_tested,
            counts.errors
        ),
        (4, 0, 0, 0)
    );
    assert!(found.is_none());
    assert!(task == Task::EMPTY);

    // The mining loop also retains full-width cursors, not only finish_tile itself.
    let count = BigUint::from(1u8) << 1000;
    let mut task = Task {
        state: 2,
        first: bytes(&first),
        count: bytes(&count),
        remaining: bytes(&count),
        cursor: bytes(&(&count - BigUint::from(2u8))),
        ..Task::EMPTY
    };
    let (counts, _) = pipeline::mine(&c, &pattern, &mut task, 0, 1, 7);
    assert_eq!((counts.q_tested, counts.errors), (7, 0));
    assert_eq!(BigUint::from_bytes_be(&task.cursor), BigUint::from(5u8));
    assert_eq!(
        BigUint::from_bytes_be(&task.remaining),
        count - BigUint::from(7u8)
    );
}

#[test]
fn mining_rejects_invalid_work_and_counter_overflow() {
    let c = config("", "");
    let pattern = logic::search::hex_pattern::HexPattern::new("", "", 256).unwrap();
    for (start, stride, steps) in [
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1025),
        (u64::MAX, 1, 2),
        (u64::MAX - 5, 3, 3),
    ] {
        let mut task = Task::EMPTY;
        let (counts, pair) = pipeline::mine(&c, &pattern, &mut task, start, stride, steps);
        assert!(task == Task::EMPTY);
        assert_eq!(counts.errors, 1);
        assert!(pair.is_none());
    }
    assert!(pipeline::launch_work(pipeline::MAX_CAPACITY, pipeline::MAX_STEPS_PER_LAUNCH).is_ok());
    assert!(pipeline::launch_work(pipeline::MAX_CAPACITY + 1, 1).is_err());
    let invalid = pipeline::Counts {
        p_tested: 64,
        q_tested: 64,
        ..Default::default()
    };
    assert!(invalid.validate(1, 64).is_err());
}

#[test]
fn eight_workers_reserve_disjoint_mining_ids() {
    use std::sync::{Arc, Barrier};
    use vanity_miner::runner::session::SearchControl;
    let control = Arc::new(SearchControl::new());
    let barrier = Arc::new(Barrier::new(8));
    let capacity = 17;
    let steps = 4;
    let work = pipeline::launch_work(capacity, steps).unwrap();
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let control = control.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                let mut ids = Vec::new();
                for _ in 0..3 {
                    let range = control.reserve_batch(u64::from(work)).unwrap();
                    for lane in 0..capacity {
                        for step in 0..steps {
                            let id = range.start + u64::from(step * capacity + lane);
                            assert!(range.contains(&id));
                            assert_eq!(id % u64::from(capacity), u64::from(lane));
                            ids.push(id);
                        }
                    }
                }
                ids
            })
        })
        .collect();
    let mut ids: Vec<_> = workers
        .into_iter()
        .flat_map(|worker| worker.join().unwrap())
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..8 * 3 * u64::from(work)).collect::<Vec<_>>());
}
