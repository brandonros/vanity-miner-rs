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

        // Arithmetic.
        arithmetic::arith_u32_div_var;
        arithmetic::arith_u32_div_const;
        arithmetic::arith_u64_div_var;
        arithmetic::arith_u64_div_const;
        arithmetic::arith_u32_rem_var;
        arithmetic::arith_u64_rem_var;
        arithmetic::arith_u32_mul_lo;
        arithmetic::arith_u64_mul_lo;
        arithmetic::arith_u64_mul_hi;
        arithmetic::arith_u128_mul;
        arithmetic::arith_overflowing_add;
        arithmetic::arith_overflowing_sub;
        arithmetic::arith_carry_chain_3limb;
        arithmetic::arith_widening_mul_pair;
        arithmetic::arith_mad_lo_u64;
        arithmetic::arith_mad_hi_u64;
        arithmetic::arith_mul_wide_u32;
        arithmetic::arith_mask_blend_true;
        arithmetic::arith_mask_blend_false;
        arithmetic::arith_var_shr_u64;
        arithmetic::arith_var_shl_u64;
        arithmetic::arith_blackbox_identity_u64;
        arithmetic::arith_blackbox_identity_u32;
        arithmetic::arith_divrem_by_58_pow_5;
        arithmetic::arith_i128_chain_add;
        arithmetic::arith_widening_mul_chain_3term;
        arithmetic::arith_u128_imm_shr_52;

        // Base58 probes.
        base58_probes::base58_var_len;
        base58_probes::base58_all_zeros;
        base58_probes::base58_div_by_58;
        base58_probes::base58_limb_divrem;
        base58_probes::base58_inner_mutate_phase;
        base58_probes::base58_min_nonzero;
        base58_probes::base58_handrolled_no_seq;

        // Layout probes.
        layout_probes::iter_static_table_lookup;
        layout_probes::iter_mut_slice_partial;
        layout_probes::iter_mut_alphabet_lookup;
        layout_probes::iter_static_slice_lookup;
        layout_probes::dynamic_index_write;
        layout_probes::static_depth4_newtype_nesting;
        layout_probes::reverse_range_write;
        layout_probes::index_trait_dispatch;
        layout_probes::named_field_struct_return;
        layout_probes::slice_reverse_partial;

        // Ed25519 probes.
        ed25519_probes::dalek_clamp_integer;
        ed25519_probes::dalek_scalar_round_trip_one;
        ed25519_probes::dalek_mul_base_scalar_one;
        ed25519_probes::dalek_scalar52_from_bytes;
        ed25519_probes::dalek_scalar52_montgomery_reduce_r;
        ed25519_probes::dalek_scalar52_mul_internal_then_reduce_one_r;
        ed25519_probes::dalek_scalar52_as_bytes_one;
        ed25519_probes::dalek_scalar52_sub_no_underflow;
        ed25519_probes::dalek_scalar52_sub_with_underflow;
        ed25519_probes::dalek_scalar52_montgomery_reduce_with_sub;
        ed25519_probes::dalek_scalar_one_to_bytes_direct;
        ed25519_probes::dalek_scalar_round_trip_zero;
        ed25519_probes::dalek_scalar_from_bytes_wide_zero;
        ed25519_probes::dalek_scalar_eq_zero;
        ed25519_probes::dalek_zero_eq_zero;
        ed25519_probes::dalek_from_canonical_zero;
        ed25519_probes::dalek_scalar52_from_bytes_zero;
        ed25519_probes::dalek_scalar52_mul_internal_zero;
        ed25519_probes::dalek_scalar52_montgomery_reduce_zero;
        ed25519_probes::dalek_scalar52_as_bytes_zero;
        ed25519_probes::dalek_reduce_pipeline_zero;
        ed25519_probes::dalek_order_boundaries;
        ed25519_probes::dalek_wide_nonzero;

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

        // Secp256k1 probes.
        secp256k1_probes::k256_secret_from_bytes_one;
        secp256k1_probes::k256_derive_scalar_one;
        secp256k1_probes::k256_derive_scalar_two;
        secp256k1_probes::k256_encode_generator;
        secp256k1_probes::k256_double_generator;
        secp256k1_probes::k256_scalar_one_round_trip;
        secp256k1_probes::k256_affine_generator_encode;
        secp256k1_probes::k256_encoded_point_from_affine_coords;
        secp256k1_probes::k256_scalar_order_boundaries;

        // Layout probes.
        layout_probes::static_u64_array_lookup;
        layout_probes::static_struct_wrapped_u64_lookup;
        layout_probes::subtle_choice_u8_into_bool;
        layout_probes::subtle_conditional_select_u64;
        layout_probes::index_trait_const_indices;
        layout_probes::generic_array_basic_index;
        layout_probes::generic_array_copy_from_slice;
        layout_probes::from_affine_coords_replica;
        layout_probes::generic_array_as_slice_last;
        layout_probes::field_bytes_into_conversion;
        layout_probes::generic_array_copy_from_ga_source;

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
    rsa_modulus ("self_test_rsa_modulus", "rsa_modulus/mod.rs") {
        // Primitives and pipeline.
        multiplication_carry;
        progression_carry;
        prime_filter;
        pseudoprime_rejected;
        zero_count_rejected;
        upper_bound_rejected;
        equal_factors_rejected;
        undersized_factor_rejected;
        end_to_end;
        device_range;
        candidate_repeatability;
        device_derivation;

        // Range probes.
        range_probes::range_empty;
        range_probes::range_multiple;
        range_probes::range_single_value;
        range_probes::range_separation;
    }
}
