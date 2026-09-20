//! Comparable transport timings for all eight modes using public test inputs.
//! Build production bundles first. Pipeline cache state is controlled by Metal;
//! `pipeline_seconds` measures this load, not a guaranteed cold compilation.
use logic::{
    crypto::{p256::public_point, sha256::Sha256},
    modes::{
        p256_public_key::P256PublicRequest, p256_signature::P256SignatureRequest,
        rsa_modulus::SearchConfig, rsa_pss::RsaPssRequest,
    },
    search::{hex_pattern::HexPattern, vanity::BytePattern, xoroshiro::BatchSeed},
};
use rsa::{BigUint, RsaPrivateKey, traits::PrivateKeyParts};
use std::{path::PathBuf, time::Instant};
use vanity_miner::modes::{
    bitcoin::metal::BitcoinTransport, ethereum::metal::EthereumTransport,
    p256_public_key::metal::P256PublicTransport, p256_signature::metal::P256SignatureTransport,
    rsa_modulus::metal::RsaTransport, rsa_pss::metal::RsaPssTransport,
    shallenge::metal::ShallengeTransport, solana::metal::SolanaTransport,
};
#[path = "../tests/support/rsa_factors.rs"]
mod factors;

fn fixed<const N: usize>(value: &BigUint) -> [u8; N] {
    let bytes = value.to_bytes_be();
    let mut out = [0; N];
    out[N - bytes.len()..].copy_from_slice(&bytes);
    out
}

// Only the measurement harness is shared; every mode's request stays explicit.
macro_rules! bench {
    ($root:expr, $audit:expr, $mode:literal, $transport:ty, $r:expr, $p:expr, $m:expr) => {{
        let mut engine = <$transport>::load(&$root.join($mode), 3, 4, $audit).unwrap();
        let (r, p, m) = ($r, $p, $m);
        for start in [0, 3] { engine.evaluate(r, p, m, start, 3).unwrap(); }
        let rounds = engine.allocation_rounds;
        let before = (engine.allocation_time, engine.upload_time, engine.download_time,
                      engine.gpu_time, engine.dispatch_time, engine.verification_time,
                      engine.cleanup_time, engine.gpu_timed_launches);
        let start = Instant::now();
        for i in 0..5 { engine.evaluate(r, p, m, 6 + i * 3, 3).unwrap(); }
        let wall = start.elapsed().as_secs_f64();
        assert_eq!(engine.allocation_rounds, rounds, "fixed capacity allocated again");
        println!("{}", serde_json::json!({
            "mode":$mode,"audit":$audit,"batch_size":3,"threads_per_group":4,
            "warmup_batches":2,"timed_batches":5,"host_seconds":wall,
            "pipeline_cache":"uncontrolled; this load only",
            "library_seconds":engine.load_stages.library.as_secs_f64(),
            "pipeline_seconds":engine.load_stages.pipeline.as_secs_f64(),
            "allocation_rounds":engine.allocation_rounds,
            "steady_allocation_rounds":engine.allocation_rounds-rounds,
            "allocation_seconds":(engine.allocation_time-before.0).as_secs_f64(),
            "upload_seconds":(engine.upload_time-before.1).as_secs_f64(),
            "download_seconds":(engine.download_time-before.2).as_secs_f64(),
            "gpu_seconds":(engine.gpu_timed_launches-before.7 == 5).then(|| (engine.gpu_time-before.3).as_secs_f64()),
            "dispatch_seconds":(engine.dispatch_time-before.4).as_secs_f64(),
            "verification_seconds":(engine.verification_time-before.5).as_secs_f64(),
            "cleanup_seconds":(engine.cleanup_time-before.6).as_secs_f64(),
        }));
    }};
}

fn main() {
    let root = std::env::var_os("VANITY_METAL_BUNDLES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/metal"));
    let seed = BatchSeed {
        seed: 42,
        width: 32,
    };
    let public = P256PublicRequest {
        seed: [0x42; 32],
        worker: 7,
        target: 0,
        reserved: 0,
    };
    let message = b"sample";
    let private = [1; 32]; // Public benchmark input, never a user's signing key.
    let signature = P256SignatureRequest {
        private,
        seed: [0x42; 32],
        fingerprint: Sha256::digest(public_point(&private).unwrap()),
        digest: Sha256::digest(message),
        worker: 3,
        offset: 0,
        length: 1,
        source: 1,
        target: 0,
        s_form: 0,
        reserved: 0,
    };
    let mut key = RsaPrivateKey::from_p_q(
        BigUint::from_bytes_be(&factors::P),
        BigUint::from_bytes_be(&factors::Q),
        BigUint::from(65537u32),
    )
    .unwrap();
    key.precompute().unwrap();
    let n = BigUint::from_bytes_be(&factors::P) * BigUint::from_bytes_be(&factors::Q);
    let mut suffix = [0; 256];
    suffix[255] = 1;
    let modulus = SearchConfig {
        lower: fixed(&n),
        upper: fixed(&n),
        p_min: factors::P,
        p_count: fixed(&BigUint::from(1u8)),
        suffix,
        seed: [42; 32],
        worker: 7,
        suffix_bits: 1,
        reserved: 0,
    };
    let pss = RsaPssRequest {
        p: fixed(&key.primes()[0]),
        q: fixed(&key.primes()[1]),
        dp: fixed(key.dp().unwrap()),
        dq: fixed(key.dq().unwrap()),
        q_inv: fixed(&key.crt_coefficient().unwrap()),
        digest: Sha256::digest(message),
        salt: [0x42; 222],
        reserved: [0; 2],
        offset: 0,
        length: 0,
        source: 0,
        salt_length: 32,
    };
    for audit in [false, true] {
        bench!(
            root,
            audit,
            "shallenge",
            ShallengeTransport,
            &seed,
            &[255; 32],
            b"benchmark"
        );
        bench!(
            root,
            audit,
            "bitcoin",
            BitcoinTransport,
            &seed,
            &BytePattern::new(b"", b"").unwrap(),
            &[]
        );
        bench!(
            root,
            audit,
            "ethereum",
            EthereumTransport,
            &seed,
            &BytePattern::new(b"", b"").unwrap(),
            &[]
        );
        bench!(
            root,
            audit,
            "solana",
            SolanaTransport,
            &seed,
            &BytePattern::new(b"", b"").unwrap(),
            &[]
        );
        bench!(
            root,
            audit,
            "p256-public-key",
            P256PublicTransport,
            &public,
            &HexPattern::new("", "", 32).unwrap(),
            &[]
        );
        bench!(
            root,
            audit,
            "p256-signature",
            P256SignatureTransport,
            &signature,
            &HexPattern::new("", "", 64).unwrap(),
            message
        );
        bench!(
            root,
            audit,
            "rsa-modulus",
            RsaTransport,
            &modulus,
            &HexPattern::new("", "", 256).unwrap(),
            &[]
        );
        bench!(
            root,
            audit,
            "rsa-pss",
            RsaPssTransport,
            &pss,
            &HexPattern::new("", "", 256).unwrap(),
            message
        );
    }
}
