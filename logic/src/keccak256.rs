use tiny_keccak::{Keccak, Hasher};

pub fn keccak256_64bytes(input: &[u8; 64]) -> [u8; 32] {
    let mut keccak = Keccak::v256();
    let mut output = [0u8; 32];
    keccak.update(input);
    keccak.finalize(&mut output);
    output
}

#[cfg(test)]
mod test {
    #[test]
    fn should_hash_correctly() {
        // sha512
        let private_key: [u8; 64] = hex::decode("61a314b0183724ea0e5f237584cb76092e253b99783d846a5b10db155128eafd61a314b0183724ea0e5f237584cb76092e253b99783d846a5b10db155128eafd").unwrap().try_into().unwrap();
        let hashed_private_key_bytes = crate::keccak256_64bytes(&private_key);
        let expected = hex::decode("0f439a9830558b9cd6842328dd11585401c34321a53b29422aacde310643d373").unwrap();
        assert_eq!(hashed_private_key_bytes, *expected);
    }
}
