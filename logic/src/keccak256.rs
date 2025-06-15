use tiny_keccak::{Keccak, Hasher};

pub fn keccak256_64bytes(input: &[u8; 64]) -> [u8; 32] {
    let mut keccak = Keccak::v256();
    let mut output = [0u8; 32];
    keccak.update(input);
    keccak.finalize(&mut output);
    output
}

pub fn keccak256_32bytes(input: &[u8; 32]) -> [u8; 32] {
    let mut keccak = Keccak::v256();
    let mut output = [0u8; 32];
    keccak.update(input);
    keccak.finalize(&mut output);
    output
}
