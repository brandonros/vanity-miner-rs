//! Explicit-salt EMSA-PSS/SHA-256 encoding, without private-key operations.
//! The mode runners restrict keys to RSA-2048; this primitive accepts the
//! RFC 8017 bit-width parameter to exercise unused-bit edge cases in tests.

use crate::crypto::sha256::Sha256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PssError {
    InvalidWidth,
    SaltTooLong,
    MaskTooLong,
}

impl core::fmt::Display for PssError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::InvalidWidth => "PSS encoded message has an invalid bit or byte width",
            Self::SaltTooLong => "PSS salt does not fit the encoded message",
            Self::MaskTooLong => "MGF1 mask exceeds its 32-bit counter capacity",
        })
    }
}

impl core::error::Error for PssError {}

/// MGF1 with SHA-256 and a big-endian 32-bit block counter.
pub fn mgf1_sha256(seed: &[u8], output: &mut [u8]) -> Result<(), PssError> {
    let blocks = output.len().div_ceil(32);
    if blocks != 0 && u32::try_from(blocks - 1).is_err() {
        return Err(PssError::MaskTooLong);
    }
    for (index, chunk) in output.chunks_mut(32).enumerate() {
        let mut hash = Sha256::new();
        hash.update(seed);
        hash.update((index as u32).to_be_bytes());
        let digest = hash.finalize();
        chunk.copy_from_slice(&digest[..chunk.len()]);
    }
    Ok(())
}

/// RFC 8017 section 9.1.1. The caller computes emBits = modulus_bits - 1.
/// All parameter checks precede output mutation. Salt bytes are copied exactly.
pub fn encode_sha256(
    message_digest: &[u8; 32],
    salt: &[u8],
    em_bits: u32,
    output: &mut [u8],
) -> Result<(), PssError> {
    let em_len = em_bits.div_ceil(8) as usize;
    if em_bits == 0 || em_len != output.len() || !(34..=256).contains(&em_len) {
        return Err(PssError::InvalidWidth);
    }
    if salt.len() > em_len - 34 {
        return Err(PssError::SaltTooLong);
    }
    let mut hash = Sha256::new();
    hash.update([0; 8]);
    hash.update(message_digest);
    hash.update(salt);
    let h = hash.finalize();
    let db_len = em_len - 33;
    mgf1_sha256(&h, &mut output[..db_len])?;
    // maskedDB = dbMask XOR (zero padding || 01 || salt).
    let delimiter = db_len - salt.len() - 1;
    output[delimiter] ^= 1;
    for (destination, byte) in output[delimiter + 1..db_len].iter_mut().zip(salt) {
        *destination ^= byte;
    }
    let unused = (8 * em_len) as u32 - em_bits;
    output[0] &= 0xff >> unused;
    output[db_len..db_len + 32].copy_from_slice(&h);
    output[em_len - 1] = 0xbc;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mgf1_partial_block_and_empty_mask() {
        let mut output = [0; 50];
        mgf1_sha256(b"public test seed", &mut output).unwrap();
        // Independently computed with Python hashlib, including the partial block.
        assert_eq!(
            hex::encode(output),
            "6d31e2b2feac188d868a4e9150de28bcbda203a729c02c06483a032f9d093795d1c5d1c7cb042c9126789c2c554dad4357d9"
        );
        let mut first = Sha256::new();
        first.update(b"public test seed");
        first.update([0; 4]);
        let first = first.finalize();
        assert_eq!(&output[..32], first.as_slice());
        let mut second = Sha256::new();
        second.update(b"public test seed");
        second.update([0, 0, 0, 1]);
        assert_eq!(&output[32..], &second.finalize()[..18]);
        mgf1_sha256(b"", &mut []).unwrap();
    }

    #[test]
    fn explicit_salt_known_answer() {
        // Independent Python hashlib encoding, for the public message "sample"
        // and salt bytes 00..1f; no RSA private material is involved.
        let digest: [u8; 32] = Sha256::digest(b"sample");
        let salt = core::array::from_fn::<_, 32, _>(|i| i as u8);
        let mut output = [0; 256];
        encode_sha256(&digest, &salt, 2047, &mut output).unwrap();
        assert_eq!(
            hex::encode(output),
            concat!(
                "4f61a706d0e288b2da2a307404bf77304e28ecd445ad93f4ed3451acee2473617",
                "828b100939cee4184f0052e70cdccfbecc74b28dc4aac818d8c8bdcf141d2209",
                "0144e097662856abfc3977c436c4a83199bbeebd9e6cbad8860d2506cb72cecd",
                "7cab414c8f9ccd9c1b295ef4e5eea57ca8d421310b231ad6ab75e8cdda5188f",
                "4ebf8340d6d5645da215878785348a9d9c314e4a450a652ce718effc9c639472",
                "77daf32f4d6b7d5553bdd9e51a83b6b9c6d146459178a5f6efa60bc2fa69025",
                "49d521edb244cfe86b61c0794866535cb1ceabd0a41c14f87e080ee349512a437",
                "4e091cebda7aa458b4e6c6ac93516915d7af61a379c5eaee3b002f683a53e7bc"
            )
        );
    }

    #[test]
    fn masking_salt_recovery_and_trailer() {
        let digest: [u8; 32] = Sha256::digest(b"sample");
        for em_bits in [2041, 2047, 2048] {
            for salt_length in [0, 1, 32, 222] {
                let salt = [0x42; 222];
                let salt = &salt[..salt_length];
                let mut output = [0; 256];
                encode_sha256(&digest, salt, em_bits, &mut output).unwrap();
                let unused = 2048 - em_bits;
                assert_eq!(output[0] & !(0xff >> unused), 0);
                assert_eq!(output[255], 0xbc);
                let mut mask = [0; 223];
                mgf1_sha256(&output[223..255], &mut mask).unwrap();
                for i in 0..223 {
                    output[i] ^= mask[i];
                }
                output[0] &= 0xff >> unused;
                let delimiter = 222 - salt_length;
                assert!(output[..delimiter].iter().all(|byte| *byte == 0));
                assert_eq!(output[delimiter], 1);
                assert_eq!(&output[delimiter + 1..223], salt);
            }
        }
    }

    #[test]
    fn invalid_parameters_do_not_modify_output() {
        let mut output = [0xa5; 256];
        assert_eq!(
            encode_sha256(&[0; 32], &[0; 223], 2047, &mut output),
            Err(PssError::SaltTooLong)
        );
        assert_eq!(output, [0xa5; 256]);
        for bits in [0, 256, 2040, 2049, u32::MAX] {
            assert_eq!(
                encode_sha256(&[0; 32], &[], bits, &mut output),
                Err(PssError::InvalidWidth)
            );
            assert_eq!(output, [0xa5; 256]);
        }
    }
}
