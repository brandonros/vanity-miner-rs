//! RustCrypto digest trait adapters for our SHA-256 implementation.
//! Used by HMAC and RSA signing; contains no hashing algorithm.
use super::sha256::Sha256;
use digest::{FixedOutput, FixedOutputReset, HashMarker, Output, OutputSizeUser, Reset, Update};
impl HashMarker for Sha256 {}
impl OutputSizeUser for Sha256 {
    type OutputSize = digest::consts::U32;
}
impl digest::core_api::BlockSizeUser for Sha256 {
    type BlockSize = digest::consts::U64;
}
impl Update for Sha256 {
    fn update(&mut self, data: &[u8]) {
        Sha256::update(self, data);
    }
}
impl FixedOutput for Sha256 {
    fn finalize_into(self, out: &mut Output<Self>) {
        out.copy_from_slice(&self.finalize());
    }
}
impl Reset for Sha256 {
    fn reset(&mut self) {
        *self = Self::new();
    }
}
impl FixedOutputReset for Sha256 {
    fn finalize_into_reset(&mut self, out: &mut Output<Self>) {
        out.copy_from_slice(&core::mem::take(self).finalize());
    }
}

#[cfg(test)]
mod tests {
    use crate::crypto::sha256::{Sha256, sha256_from_bytes};
    #[test]
    fn digest_reset_and_cloned_prefixes() {
        let mut hash = Sha256::new();
        hash.update(b"shared prefix:");
        let mut other = hash.clone();
        hash.update(b"a");
        other.update(b"b");
        assert_eq!(hash.finalize(), sha256_from_bytes(b"shared prefix:a"));
        let result = digest::Digest::finalize_reset(&mut other);
        assert_eq!(&result[..], &sha256_from_bytes(b"shared prefix:b"));
        assert_eq!(other.clone().finalize(), sha256_from_bytes(b""));
        other.update(b"discard this");
        digest::Reset::reset(&mut other);
        other.update(b"abc");
        assert_eq!(
            hex::encode(other.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[cfg(any(feature = "p256-public-key", feature = "p256-signature"))]
    #[test]
    fn rfc4231_hmac_short_and_long_keys() {
        use hmac::{Mac, SimpleHmac};
        let mut mac = SimpleHmac::<Sha256>::new_from_slice(&[0x0b; 20]).unwrap();
        mac.update(b"Hi There");
        assert_eq!(
            hex::encode(mac.finalize().into_bytes()),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        let mut mac = SimpleHmac::<Sha256>::new_from_slice(&[0xaa; 131]).unwrap();
        mac.update(b"Test Using Larger Than Block-Size Key - Hash Key First");
        assert_eq!(
            hex::encode(mac.finalize().into_bytes()),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }
}
