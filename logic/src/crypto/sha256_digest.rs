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
