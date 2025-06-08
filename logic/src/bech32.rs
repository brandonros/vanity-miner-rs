// Bech32 character set
const BECH32_ALPHABET: &[u8; 32] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";

// Generator polynomial for checksum
const BECH32_CONST: u32 = 1;
const GENERATORS: [u32; 5] = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];

/// Convert 8-bit data to 5-bit groups
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

/// Polymod function for checksum calculation
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

/// Calculate Bech32 checksum
fn calculate_checksum(hrp: &[u8], data: &[u8]) -> [u8; 6] {
    let mut values = [0u8; 256]; // Max reasonable size
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

    let mod_val = polymod(&values[..values_len]) ^ BECH32_CONST;
    let mut checksum = [0u8; 6];
    for i in 0..6 {
        checksum[i] = ((mod_val >> (5 * (5 - i))) & 31) as u8;
    }
    checksum
}

/// Encode data as Bech32
pub fn bech32_encode(hrp: &[u8], data: &[u8], output: &mut [u8]) -> usize {
    let checksum = calculate_checksum(hrp, data);
    
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

/// Encode Bitcoin P2WPKH address (bc1q...)
pub fn encode_p2wpkh_address(pubkey_hash: &[u8; 20], mainnet: bool, output: &mut [u8]) -> usize {
    let hrp = if mainnet { b"bc" } else { b"tb" };
    
    // Create witness program: version (0) + pubkey_hash converted to 5-bit
    let mut witness_program = [0u8; 64];
    witness_program[0] = 0; // Version 0 for P2WPKH
    
    let converted = convert_bits(pubkey_hash, 8, 5, true);
    
    // Find actual length of converted data
    let converted_len = 32;
    
    // Copy converted data
    for i in 0..converted_len {
        witness_program[i + 1] = converted[i];
    }
    
    bech32_encode(hrp, &witness_program[..converted_len + 1], output)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn should_encode_correctly() {
        let public_key_hash: [u8; 20] = hex::decode("46047c8a3d8edb134c3f1a3e7d65b0fd7421f127").unwrap().try_into().unwrap();
        let mut encoded_public_key = [0u8; 64];
        let encoded_len = encode_p2wpkh_address(&public_key_hash, true, &mut encoded_public_key);
        let encoded_public_key = &encoded_public_key[0..encoded_len];
        let expected = b"bc1qgcz8ez3a3md3xnplrgl86edsl46zruf8mwx56m";
        assert_eq!(*encoded_public_key, *expected);
    }
}
