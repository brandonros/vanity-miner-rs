//! File names for independently compiled production kernels.
pub fn production_module(kernel: &str) -> &'static str {
    match kernel {
        "kernel_solana_vanity" => "solana",
        "kernel_bitcoin_vanity" => "bitcoin",
        "kernel_ethereum_vanity" => "ethereum",
        "kernel_shallenge" => "shallenge",
        "kernel_p256_public_key_vanity" => "p256_public_key",
        "kernel_p256_signature_vanity" => "p256_signature",
        "kernel_rsa_pss_signature_vanity" => "rsa_pss",
        "kernel_rsa_generate"
        | "kernel_rsa_ranges"
        | "kernel_rsa_search"
        | "kernel_rsa_advance" => "rsa_modulus",
        _ => panic!("unknown production kernel: {kernel}"),
    }
}

/// Standalone PTX containing this entry.
pub fn self_test_module(kernel: &str) -> &str {
    kernel.strip_prefix("kernel_").expect("kernel entry prefix")
}
