//! Host support for bounded cryptographic searches.

#[cfg(feature = "p256-public-key")]
pub mod p256_public;
#[cfg(feature = "p256-signature")]
pub mod p256_signature;
#[cfg(feature = "rsa-common")]
pub mod rsa_host;
#[cfg(all(test, feature = "rsa-pss"))]
mod rsa_interop;
#[cfg(feature = "rsa-modulus")]
pub mod rsa_modulus;
#[cfg(feature = "rsa-pss")]
pub mod rsa_pss_search;
pub mod search_control;

#[cfg(any(
    feature = "rsa-modulus",
    feature = "rsa-pss",
    feature = "p256-public-key",
    feature = "p256-signature"
))]
pub mod search_batches;

#[cfg(all(
    any(test, feature = "self_test_support"),
    any(
        feature = "p256-public-key",
        feature = "p256-signature",
        feature = "rsa-modulus",
        feature = "rsa-pss"
    )
))]
pub mod test_support;

pub mod stats;

#[cfg(feature = "self_test_support")]
pub mod self_test_suite;

pub mod kernel_modules;
