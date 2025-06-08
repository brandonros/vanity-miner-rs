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
