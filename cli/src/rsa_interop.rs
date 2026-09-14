//! Independent host tests for shared fixed-width RSA arithmetic.

use logic::{crypto::rsa_crt::Rsa2048Crt, crypto::rsa_pss::encode_sha256};
use rand::{RngCore, rngs::OsRng};
use rsa::{
    BigUint, Pss, RsaPrivateKey, RsaPublicKey,
    pkcs8::{EncodePublicKey, LineEnding},
    traits::{PrivateKeyParts, PublicKeyParts},
};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

fn fixed<const N: usize>(value: &BigUint) -> Zeroizing<[u8; N]> {
    let bytes = Zeroizing::new(value.to_bytes_be());
    assert!(bytes.len() <= N);
    let mut output = Zeroizing::new([0; N]);
    output[N - bytes.len()..].copy_from_slice(&bytes);
    output
}

struct Directory(std::path::PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn generated_rsa2048_crt_pss_matches_independent_verifiers() {
    // Generate fresh private material in memory; no private fixture or logging.
    let mut key = RsaPrivateKey::new(&mut OsRng, 2048).unwrap();
    key.validate().unwrap();
    key.precompute().unwrap();
    assert_eq!(key.n().bits(), 2048);
    assert_eq!(key.primes().len(), 2);
    assert!(key.primes().iter().all(|prime| prime.bits() == 1024));
    let p = fixed(&key.primes()[0]);
    let q = fixed(&key.primes()[1]);
    let dp = fixed(key.dp().unwrap());
    let dq = fixed(key.dq().unwrap());
    let coefficient = Zeroizing::new(key.crt_coefficient().unwrap());
    let q_inv = fixed(&coefficient);
    let crt = Rsa2048Crt::new(&p, &q, &dp, &dq, &q_inv).unwrap();
    assert_eq!(crt.modulus().as_slice(), key.n().to_bytes_be());
    let public = RsaPublicKey::from(&key);
    let message = b"independent RSA-2048 explicit-salt PSS test";
    let digest: [u8; 32] = Sha256::digest(message).into();
    for length in [0, 1, 32, 222] {
        let salt: Vec<u8> = (0..length).map(|i| i as u8).collect();
        let mut encoded = [0; 256];
        encode_sha256(&digest, &salt, 2047, &mut encoded).unwrap();
        let signature = crt.private_operation(&encoded).unwrap();
        // rsa uses num-bigint-dig, independently of shared crypto-bigint math.
        public
            .verify(Pss::new_with_salt::<Sha256>(length), &digest, &signature)
            .unwrap();
        assert!(
            public
                .verify(
                    Pss::new_with_salt::<Sha256>(length + 1),
                    &digest,
                    &signature
                )
                .is_err()
        );
        if length == 32
            && std::process::Command::new("openssl")
                .arg("version")
                .output()
                .is_ok()
        {
            let mut random = [0; 16];
            OsRng.fill_bytes(&mut random);
            let dir = Directory(
                std::env::temp_dir().join(format!("vanity-rsa-interop-{}", hex::encode(random))),
            );
            std::fs::create_dir(&dir.0).unwrap();
            let public_path = dir.0.join("public.pem");
            let message_path = dir.0.join("message.bin");
            let signature_path = dir.0.join("signature.bin");
            std::fs::write(
                &public_path,
                public.to_public_key_pem(LineEnding::LF).unwrap(),
            )
            .unwrap();
            std::fs::write(&message_path, message).unwrap();
            std::fs::write(&signature_path, signature).unwrap();
            let result = std::process::Command::new("openssl")
                .args(["dgst", "-sha256", "-verify"])
                .arg(&public_path)
                .arg("-signature")
                .arg(&signature_path)
                .args([
                    "-sigopt",
                    "rsa_padding_mode:pss",
                    "-sigopt",
                    "rsa_pss_saltlen:32",
                ])
                .arg(&message_path)
                .output()
                .unwrap();
            assert!(result.status.success());
        }
    }
}
