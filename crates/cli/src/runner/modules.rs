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
        _ => panic!("unknown production kernel: {kernel}"),
    }
}
