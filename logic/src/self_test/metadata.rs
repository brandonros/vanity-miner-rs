//! Stable self-test ownership and labels. No source parsing or GPU function pointers.

#[derive(Clone, Copy)]
pub struct Case {
    pub slot: usize,
    pub label: &'static str,
    pub kernel: &'static str,
    pub gpu_skip: Option<&'static str>,
}
#[path = "bitcoin/cases.rs"]
mod bitcoin;
#[path = "ethereum/cases.rs"]
mod ethereum;
#[path = "p256_public_key/cases.rs"]
mod p256_public_key;
#[path = "p256_signature/cases.rs"]
mod p256_signature;
#[path = "rsa_modulus/cases.rs"]
mod rsa_modulus;
#[path = "rsa_pss/cases.rs"]
mod rsa_pss;
#[path = "shallenge/cases.rs"]
mod shallenge;
#[path = "solana/cases.rs"]
mod solana;

pub const GROUPS: &[(&[Case], bool)] = &[
    (solana::CASES, cfg!(feature = "self_test_solana")),
    (bitcoin::CASES, cfg!(feature = "self_test_bitcoin")),
    (ethereum::CASES, cfg!(feature = "self_test_ethereum")),
    (shallenge::CASES, cfg!(feature = "self_test_shallenge")),
    (
        p256_public_key::CASES,
        cfg!(feature = "self_test_p256_public_key"),
    ),
    (
        p256_signature::CASES,
        cfg!(feature = "self_test_p256_signature"),
    ),
    (rsa_pss::CASES, cfg!(feature = "self_test_rsa_pss")),
    (rsa_modulus::CASES, cfg!(feature = "self_test_rsa_modulus")),
];

pub const fn labels() -> [&'static str; super::SELF_TEST_NUM_CHECKS] {
    let mut labels = [""; super::SELF_TEST_NUM_CHECKS];
    let mut g = 0;
    while g < GROUPS.len() {
        let cases = GROUPS[g].0;
        let mut i = 0;
        while i < cases.len() {
            labels[cases[i].slot] = cases[i].label;
            i += 1;
        }
        g += 1;
    }
    labels
}
