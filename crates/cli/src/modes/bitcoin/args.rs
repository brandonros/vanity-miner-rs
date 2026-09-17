use std::error::Error;

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
    let separator_pos = bech32_lower.rfind('1').ok_or("Missing '1' separator")?;

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

/// Suffixes contain only the Bech32 data alphabet, with no address HRP/separator.
pub fn validate_bech32_suffix(suffix: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    for c in suffix.chars() {
        if !BECH32_CHARSET.contains(c) {
            return Err(format!("Invalid Bitcoin suffix character: '{c}'").into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitcoin_suffix_is_data_without_address_prefix() {
        for suffix in ["", "qq", "ff", "0239"] {
            assert!(validate_bech32_suffix(suffix).is_ok());
        }
        for suffix in ["bc1q", "1", "b", "i", "o", "Q", " "] {
            assert!(validate_bech32_suffix(suffix).is_err());
        }
    }
}

#[derive(clap::Args, Clone)]
pub struct BitcoinArgs {
    /// Prefix to search for
    #[arg(long, default_value = "")]
    pub prefix: String,
    /// Suffix to search for
    #[arg(long, default_value = "")]
    pub suffix: String,
}

impl BitcoinArgs {
    pub fn validate(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let Self { prefix, suffix } = self;

        if !prefix.is_empty() {
            validate_bech32_string(prefix)?;
        }
        if !suffix.is_empty() {
            validate_bech32_suffix(suffix)?;
        }

        Ok(())
    }
    pub fn details(&self) -> crate::args::CommandDetails {
        let Self { prefix, suffix } = self;
        crate::args::CommandDetails {
            prefix_len: prefix.len(),
            suffix_len: suffix.len(),
            cpu_threads: None,
            cuda_module: Some("bitcoin"),
            description: format!(
                "Searching for bitcoin vanity key with prefix '{}' and suffix '{}'",
                prefix, suffix
            ),
        }
    }
}
