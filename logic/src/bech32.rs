// Bech32 character set (unchanged)
const BECH32_ALPHABET: &[u8; 32] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";

// Generator polynomial for checksum (unchanged)
const BECH32_CONST: u32 = 1;
const BECH32M_CONST: u32 = 0x2bc830a3; // NEW: Bech32m constant
const GENERATORS: [u32; 5] = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];

#[derive(Clone, Copy)]
pub enum Bech32Variant {
    Bech32,
    Bech32m,
}

/// Convert 8-bit data to 5-bit groups (unchanged)
fn convert_bits(data: &[u8], from_bits: u8, to_bits: u8, pad: bool) -> [u8; 64] {
    let mut result = [0u8; 64];
    let mut result_len = 0;
    let mut acc = 0u32;
    let mut bits = 0u8;
    let maxv = (1u32 << to_bits) - 1;
    let max_acc = (1u32 << (from_bits + to_bits - 1)) - 1;

    for &value in data {
        acc = ((acc << from_bits) | (value as u32)) & max_acc;
        bits += from_bits;
        while bits >= to_bits {
            bits -= to_bits;
            result[result_len] = ((acc >> bits) & maxv) as u8;
            result_len += 1;
        }
    }

    if pad {
        if bits > 0 {
            result[result_len] = ((acc << (to_bits - bits)) & maxv) as u8;
            result_len += 1;
        }
    }

    // Zero out unused portion
    for i in result_len..result.len() {
        result[i] = 0;
    }

    result
}

/// Polymod function for checksum calculation (unchanged)
fn polymod(values: &[u8]) -> u32 {
    let mut chk = 1u32;
    for &value in values {
        let top = chk >> 25;
        chk = (chk & 0x1ffffff) << 5 ^ (value as u32);
        for i in 0..5 {
            chk ^= if (top >> i) & 1 == 1 { GENERATORS[i] } else { 0 };
        }
    }
    chk
}

/// Calculate checksum (MODIFIED to support both variants)
fn calculate_checksum(hrp: &[u8], data: &[u8], variant: Bech32Variant) -> [u8; 6] {
    let mut values = [0u8; 256];
    let mut values_len = 0;

    // Add HRP
    for &c in hrp {
        if values_len >= values.len() {
            break;
        }
        values[values_len] = c >> 5;
        values_len += 1;
    }
    
    if values_len < values.len() {
        values[values_len] = 0;
        values_len += 1;
    }
    
    for &c in hrp {
        if values_len >= values.len() {
            break;
        }
        values[values_len] = c & 31;
        values_len += 1;
    }

    // Add data
    for &d in data {
        if values_len >= values.len() {
            break;
        }
        values[values_len] = d;
        values_len += 1;
    }

    // Add 6 zeros for checksum
    for _ in 0..6 {
        if values_len >= values.len() {
            break;
        }
        values[values_len] = 0;
        values_len += 1;
    }

    // Use appropriate constant based on variant
    let const_val = match variant {
        Bech32Variant::Bech32 => BECH32_CONST,
        Bech32Variant::Bech32m => BECH32M_CONST,
    };

    let mod_val = polymod(&values[..values_len]) ^ const_val;
    let mut checksum = [0u8; 6];
    for i in 0..6 {
        checksum[i] = ((mod_val >> (5 * (5 - i))) & 31) as u8;
    }
    checksum
}

/// Encode data as Bech32/Bech32m (MODIFIED)
pub fn bech32_encode(hrp: &[u8], data: &[u8], variant: Bech32Variant, output: &mut [u8]) -> usize {
    let checksum = calculate_checksum(hrp, data, variant);
    
    let mut pos = 0;
    
    // Add HRP
    for &c in hrp {
        output[pos] = c;
        pos += 1;
    }
    
    // Add separator
    output[pos] = b'1';
    pos += 1;
    
    // Add data
    for &d in data {
        output[pos] = BECH32_ALPHABET[d as usize];
        pos += 1;
    }
    
    // Add checksum
    for &c in &checksum {
        output[pos] = BECH32_ALPHABET[c as usize];
        pos += 1;
    }
    
    pos
}

/// Encode Bitcoin P2WPKH address (bc1q...) - Segwit v0
pub fn encode_p2wpkh_address(pubkey_hash: &[u8; 20], mainnet: bool, output: &mut [u8]) -> usize {
    let hrp = if mainnet { b"bc" } else { b"tb" };
    
    let mut witness_program = [0u8; 64];
    witness_program[0] = 0; // Version 0
    
    let converted = convert_bits(pubkey_hash, 8, 5, true);
    let converted_len = 32;
    
    for i in 0..converted_len {
        witness_program[i + 1] = converted[i];
    }
    
    // Segwit v0 uses Bech32
    bech32_encode(hrp, &witness_program[..converted_len + 1], Bech32Variant::Bech32, output)
}

/// NEW: Encode Bitcoin P2TR address (bc1p...) - Segwit v1 (Taproot)
pub fn encode_p2tr_address(pubkey: &[u8; 32], mainnet: bool, output: &mut [u8]) -> usize {
    let hrp = if mainnet { b"bc" } else { b"tb" };
    
    let mut witness_program = [0u8; 64];
    witness_program[0] = 1; // Version 1 for Taproot
    
    let converted = convert_bits(pubkey, 8, 5, true);
    let converted_len = 52; // 32 bytes -> 52 5-bit groups (with padding)
    
    for i in 0..converted_len {
        witness_program[i + 1] = converted[i];
    }
    
    // Segwit v1+ uses Bech32m
    bech32_encode(hrp, &witness_program[..converted_len + 1], Bech32Variant::Bech32m, output)
}

/// NEW: Generic witness program encoder
pub fn encode_witness_program(version: u8, program: &[u8], mainnet: bool, output: &mut [u8]) -> usize {
    let hrp = if mainnet { b"bc" } else { b"tb" };
    
    let mut witness_program = [0u8; 64];
    witness_program[0] = version;
    
    let converted = convert_bits(program, 8, 5, true);
    // Calculate actual converted length based on program length
    let converted_len = (program.len() * 8 + 4) / 5;
    
    for i in 0..converted_len {
        witness_program[i + 1] = converted[i];
    }
    
    // Use Bech32 for v0, Bech32m for v1+
    let variant = if version == 0 {
        Bech32Variant::Bech32
    } else {
        Bech32Variant::Bech32m
    };
    
    bech32_encode(hrp, &witness_program[..converted_len + 1], variant, output)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn should_encode_p2wpkh_correctly() {
        let public_key_hash: [u8; 20] = hex::decode("46047c8a3d8edb134c3f1a3e7d65b0fd7421f127").unwrap().try_into().unwrap();
        let mut encoded_public_key = [0u8; 64];
        let encoded_len = encode_p2wpkh_address(&public_key_hash, true, &mut encoded_public_key);
        let encoded_public_key = &encoded_public_key[0..encoded_len];
        let expected = b"bc1qgcz8ez3a3md3xnplrgl86edsl46zruf8mwx56m";
        assert_eq!(*encoded_public_key, *expected);
    }

    #[test]
    fn should_encode_p2tr_correctly() {
        let public_key: [u8; 32] = hex::decode("46047c8a3d8edb134c3f1a3e7d65b0fd7421f127ff3355433344445553111444").unwrap().try_into().unwrap();
        let mut encoded_public_key = [0u8; 64];
        let encoded_len = encode_p2tr_address(&public_key, true, &mut encoded_public_key);
        let encoded_public_key = &encoded_public_key[0..encoded_len];
        let expected = b"bc1pgcz8ez3a3md3xnplrgl86edsl46zruf8lue42seng3z925c3z3zqc5wf5u";
        assert_eq!(*encoded_public_key, *expected);
    }
}