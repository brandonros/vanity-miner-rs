#![no_std]

extern crate alloc;

mod vanity;
mod xoroshiro;
mod base58;
mod ed25519;
mod secp256k1;
mod sha256;
mod sha512;
mod ripemd160;
mod shallenge;

pub use vanity::*;
pub use xoroshiro::*;
pub use base58::*;
pub use ed25519::*;
pub use secp256k1::*;
pub use sha256::*;
pub use sha512::*;
pub use ripemd160::*;
pub use shallenge::*;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn should_hash_correctly() {
        // sha512
        let private_key: [u8; 32] = hex::decode("61a314b0183724ea0e5f237584cb76092e253b99783d846a5b10db155128eafd").unwrap().try_into().unwrap();
        let hashed_private_key_bytes = sha512::sha512_32bytes_from_bytes(&private_key);
        let expected = hex::decode("152d53723da4203478574b153143a7eaa921a8d82c629517d6b18949f0111abb0f5b8817a8e43510f83333417178f2f59fdc3c723199303a5f9be71af2f7b664").unwrap();
        assert_eq!(hashed_private_key_bytes, *expected);

        // derive public key
        let public_key_bytes = ed25519::ed25519_derive_public_key(&hashed_private_key_bytes);
        let expected = hex::decode("0af764c1b6133a3a0abd7ef9c853791b687ce1e235f9dc8466d886da314dbea7").unwrap();
        assert_eq!(public_key_bytes, *expected);

        // bs58 encode public key
        let mut bs58_encoded_public_key = [0u8; 64];
        let encoded_len = base58::base58_encode(&public_key_bytes, &mut bs58_encoded_public_key);
        let bs58_encoded_public_key = &bs58_encoded_public_key[0..encoded_len];
        let expected = hex::decode("6a6f7365413875757746426a58707558423879453233437845756d596758336a486251677753627166504c").unwrap();
        assert_eq!(*bs58_encoded_public_key, *expected);

        // utf8
        use alloc::string::String;
        let bs58_encoded_public_key_string = String::from_utf8(bs58_encoded_public_key.to_vec()).unwrap();
        let expected = "joseA8uuwFBjXpuXB8yE23CxEumYgX3jHbQgwSbqfPL";
        assert_eq!(bs58_encoded_public_key_string, expected);
    }
}
