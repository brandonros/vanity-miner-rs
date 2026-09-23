// Registration order and mode ownership. Definitions own names and descriptions.
// Add a register_self_test! definition and list its path here.
// Every entry participates in indexing, regardless of enabled features.
define_self_tests! {
    solana ("self_test_solana", "solana/mod.rs") {
        // Primitives and pipeline.
        primitive_xoroshiro;
        primitive_sha512;
        primitive_ed25519;
        primitive_base58;
        private_key;
        public_key;
        encoded;

        // Batch seed probes.
        batch_seed_probes::batch_seed_boundary;
        batch_seed_probes::batch_seed_invalid_width;

        // Candidate probes.
        candidate_probes::candidate_match;
        candidate_probes::candidate_miss;
        candidate_probes::candidate_invalid;
    }
    bitcoin ("self_test_bitcoin", "bitcoin/mod.rs") {
        // Primitives and pipeline.
        primitive_secp256k1_compressed;
        primitive_ripemd160;
        primitive_sha256_32;
        private_key;
        public_key;
        pkh;
        encoded;
        matches;
        wif_compressed_mainnet;
        wif_uncompressed_mainnet;
        wif_compressed_testnet;
        wif_uncompressed_testnet;
        bech32_p2wpkh;

        // Base58 probes.
        base58_probes::base58_var_len_leading_zero;

        // Matching probes.
        matching_probes::vanity_prefix_suffix;
        matching_probes::vanity_length_boundaries;

        // Candidate probes.
        candidate_probes::candidate_match;
        candidate_probes::candidate_miss;
        candidate_probes::candidate_invalid;
    }
    ethereum ("self_test_ethereum", "ethereum/mod.rs") {
        // Primitives and pipeline.
        primitive_secp256k1_uncompressed;
        primitive_keccak256;
        private_key;
        public_key;
        address;

        // Candidate probes.
        candidate_probes::candidate_match;
        candidate_probes::candidate_miss;
        candidate_probes::candidate_invalid;
    }
    shallenge ("self_test_shallenge", "shallenge/mod.rs") {
        // Primitives and pipeline.
        primitive_sha256_variable;
        hash;
        nonce_len;
        is_better;
        compare_hashes_lt;
        compare_hashes_gt;
        compare_hashes_eq;
        xoroshiro_base64_nonce;

        // Sha256 probes.
        sha256_probes::sha256_padding_0;
        sha256_probes::sha256_padding_55;
        sha256_probes::sha256_padding_56;
        sha256_probes::sha256_padding_63;
        sha256_probes::sha256_padding_64;
        sha256_probes::sha256_padding_65;
        sha256_probes::sha256_streaming_boundary;
        sha256_probes::sha256_multiblock;
        sha256_probes::sha256_streaming_chunks;

        // Comparison probes.
        comparison_probes::compare_hashes_last_byte;

        // Candidate probes.
        candidate_probes::candidate_match;
        candidate_probes::candidate_miss;
        candidate_probes::candidate_invalid;
    }
    p256_public_key ("self_test_p256_public_key", "p256_public_key/mod.rs") {
        // Primitives and pipeline.
        hmac_derivation;
        scalar_derivation;
        generator;
        point_double;
        zero_scalar_rejected;
        order_scalar_rejected;
        x_encoding;
        y_encoding;
        end_to_end;

        // Matching probes.
        matching_probes::hex_pattern_nibbles;
        matching_probes::hex_pattern_max_width;

        // Scalar probes.
        scalar_probes::order_minus_one;
        scalar_probes::above_order;
    }
    p256_signature ("self_test_p256_signature", "p256_signature/mod.rs") {
        // Primitives and pipeline.
        rfc6979_sample;
        rfc6979_test;
        ephemeral_r;
        ephemeral_signature;
        zero_nonce_rejected;
        low_s;
        high_s;
        message_window_carry;
        ephemeral_hmac;
        end_to_end;

        // Window probes.
        window_probes::message_counter_exhaustion;
        window_probes::message_window_invalid;
        window_probes::message_window_multiblock;
    }
    rsa_pss ("self_test_rsa_pss", "rsa_pss/mod.rs") {
        // Primitives and pipeline.
        sha256;
        mgf1_partial_block;
        salt32_encoding;
        empty_salt_encoding;
        maximum_salt_encoding;
        oversized_salt_rejected;
        salt_carry;
        crt_known_answer;
        crt_fault_rejected;
        crt_modulus_rejected;
        end_to_end;

        // Salt probes.
        salt_probes::salt_counter_exhaustion;
        salt_probes::salt_empty_and_invalid;
        salt_probes::salt_carry_beyond_u64;
    }
}
