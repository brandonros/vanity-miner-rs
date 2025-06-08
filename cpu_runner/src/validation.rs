use std::error::Error;

pub fn validate_base58_string(base58_string: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    let invalid_characters = ["l", "I", "0", "O"];
    for invalid_character in invalid_characters {
        if base58_string.contains(invalid_character) {
            return Err(format!("base58 string contains invalid character: {}", invalid_character).into());
        }
    }
    Ok(())
}

pub fn validate_hex_string(hex_string: &str) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
    if hex_string.len() != 64 {
        return Err("Hash must be 64 hex characters (32 bytes)".into());
    }
    hex::decode(hex_string).map_err(|e| format!("Invalid hex string: {}", e).into())
}

const BECH32_CHARSET: &str = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

pub fn validate_bech32_string(bech32_string: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Check for mixed case
    let has_lower = bech32_string.chars().any(|c| c.is_ascii_lowercase());
    let has_upper = bech32_string.chars().any(|c| c.is_ascii_uppercase());
    if has_lower && has_upper {
        return Err("Mixed case not allowed in bech32".into());
    }

    let bech32_lower = bech32_string.to_ascii_lowercase();

    // Find separator
    let separator_pos = bech32_lower.rfind('1')
        .ok_or("Missing '1' separator")?;

    let hrp = &bech32_lower[..separator_pos];
    let data = &bech32_lower[separator_pos + 1..];

    // Validate HRP
    if hrp.is_empty() {
        return Err("Empty HRP".into());
    }

    // Validate data part characters
    for c in data.chars() {
        if !BECH32_CHARSET.contains(c) {
            return Err(format!("Invalid character: '{}'", c).into());
        }
    }

    // For Bitcoin, check HRP
    if hrp != "bc" && hrp != "tb" {
        return Err("Invalid Bitcoin HRP (must be 'bc' or 'tb')".into());
    }

    Ok(())
}
