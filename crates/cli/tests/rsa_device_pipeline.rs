#![cfg(feature = "rsa-modulus")]
use logic::modes::rsa_modulus::{self as pipeline, SearchConfig};
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

#[path = "support/rsa_factors.rs"]
mod factors;
use logic::search::{
    candidate_result::{BatchResult, CandidateResult},
    hex_pattern::HexPattern,
};
fn known_config() -> SearchConfig {
    let n = BigUint::from_bytes_be(&factors::P) * BigUint::from_bytes_be(&factors::Q);
    let mut c = config(&hex::encode(bytes::<256>(&n)), "");
    c.p_min = factors::P;
    c.p_count = bytes(&BigUint::from(1u8));
    c
}
#[test]
fn independent_candidate_matches_and_export_passes_host_verification() {
    use rsa::{RsaPrivateKey, pkcs8::DecodePrivateKey, traits::PublicKeyParts};
    use vanity_miner::modes::rsa_modulus::device::verify_pair;
    let c = known_config();
    let pattern = HexPattern::new("", "", 256).unwrap();
    for id in [9, 10, 9, u64::MAX] {
        let result = pipeline::rsa_modulus(&c, id, &pattern);
        assert_eq!(result.status, 1);
        assert_eq!(result.bytes[..128], factors::P);
        assert_eq!(result.bytes[128..], factors::Q);
    }
    let pair = pipeline::Pair {
        p: factors::P,
        q: factors::Q,
        id: 9,
    };
    let output = zeroize::Zeroizing::new(verify_pair(&c, &pattern, &pair).unwrap());
    let encoded = output
        .lines()
        .find_map(|l| l.strip_prefix("[rsa-modulus] private_key_pkcs8="))
        .unwrap();
    let der = zeroize::Zeroizing::new(hex::decode(encoded).unwrap());
    let key = RsaPrivateKey::from_pkcs8_der(&der).unwrap();
    key.validate().unwrap();
    assert_eq!(bytes::<256>(key.n()), c.lower);
    let mut corrupt = pair;
    corrupt.p[1] ^= 1;
    assert!(verify_pair(&c, &pattern, &corrupt).is_err());
    let mut corrupt = pair;
    corrupt.q[1] ^= 1;
    assert!(verify_pair(&c, &pattern, &corrupt).is_err());
    assert_eq!(
        pipeline::rsa_modulus(&c, 9, &HexPattern::new("00", "", 256).unwrap()).status,
        0
    );
}
#[test]
fn sampling_excludes_close_factors_and_is_independent_of_batch_partition() {
    let mut c = config("", "1");
    let p = BigUint::from_bytes_be(&factors::P);
    let d = BigUint::from(1u8) << 924usize;
    // A narrow progression straddles both boundaries of the excluded region.
    c.lower = bytes::<256>(&(&p * (&p - &d - BigUint::from(64u8))));
    c.upper = bytes::<256>(&(&p * (&p + &d + BigUint::from(64u8))));
    let baseline: Vec<_> = (0..65)
        .map(|id| pipeline::generate_q(&c, &factors::P, id).unwrap().unwrap())
        .collect();
    for chunk in [1, 7, 32, 64] {
        for start in (0..65).step_by(chunk) {
            for id in start..(start + chunk).min(65) {
                let q = pipeline::generate_q(&c, &factors::P, id as u64)
                    .unwrap()
                    .unwrap();
                assert_eq!(q, baseline[id]);
                let q = BigUint::from_bytes_be(&q);
                let distance = if q > p { &q - &p } else { &p - &q };
                assert!(distance > d);
                assert_eq!((&p * q) % 16u8, BigUint::from(1u8));
            }
        }
    }
    assert!(
        baseline
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1
    );
    c.lower = bytes::<256>(&(&p * &p));
    c.upper = c.lower;
    c.suffix = c.lower;
    c.suffix_bits = 2048;
    assert!(pipeline::generate_q(&c, &factors::P, 9).unwrap().is_none());
}
#[test]
fn malformed_requests_are_errors_and_empty_ranges_are_misses() {
    let c = known_config();
    let pattern = HexPattern::new("", "", 256).unwrap();
    for invalid in [
        SearchConfig {
            suffix_bits: 0,
            ..c
        },
        SearchConfig {
            suffix_bits: 2049,
            ..c
        },
        SearchConfig { reserved: 1, ..c },
        SearchConfig {
            p_count: [0; 128],
            ..c
        },
        SearchConfig {
            upper: [0; 256],
            ..c
        },
    ] {
        assert_eq!(
            pipeline::rsa_modulus(&invalid, 9, &pattern).status,
            CandidateResult::STATUS_ERROR
        );
    }
    let mut empty = c;
    let n = BigUint::from_bytes_be(&c.lower) + BigUint::from(1u8);
    empty.lower = bytes(&n);
    empty.upper = empty.lower;
    assert_eq!(
        pipeline::rsa_modulus(&empty, 9, &pattern).status,
        CandidateResult::STATUS_MISS
    );
}
#[test]
fn shared_batch_validation_rejects_bad_winners_and_propagates_errors() {
    use vanity_miner::{
        modes::rsa_modulus::{ModulusSearch, device as host},
        runner::session::SearchControl,
    };
    let search = ModulusSearch {
        prefix: "abc".into(),
        suffix: String::new(),
        workers: 1,
    };
    for output in [
        BatchResult {
            matches: 1,
            errors: 0,
            lane: 65,
            candidate: CandidateResult::matched(&[0; 256]),
        },
        BatchResult {
            errors: 1,
            ..BatchResult::EMPTY
        },
    ] {
        let control = SearchControl::new();
        let result = host::run(&search, &control, |_, _, _, _| Ok(output));
        assert!(result.is_err());
        assert!(control.stopped());
    }
}
