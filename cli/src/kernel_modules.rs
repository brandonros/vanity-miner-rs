//! File names for independently compiled production kernels.
pub fn production_module(kernel: &str) -> &'static str {
    match kernel {
        "kernel_find_solana_vanity_private_key" => "solana",
        "kernel_find_bitcoin_vanity_private_key" => "bitcoin",
        "kernel_find_ethereum_vanity_private_key" => "ethereum",
        "kernel_find_better_shallenge_nonce" => "shallenge",
        "kernel_p256_public_key_vanity" => "p256_public_key",
        "kernel_p256_signature_vanity" => "p256_signature",
        "kernel_rsa_pss_signature_vanity" => "rsa_pss",
        "kernel_rsa_modulus_vanity_v2" => "rsa_modulus",
        _ => panic!("unknown production kernel: {kernel}"),
    }
}
