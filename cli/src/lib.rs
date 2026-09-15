//! Host support for bounded cryptographic searches.

pub mod device_workers;
#[cfg(feature = "rsa-common")]
pub mod rsa_host;
#[cfg(all(test, feature = "rsa-pss"))]
mod rsa_interop;
pub mod search_control;

#[cfg(any(
    feature = "rsa-modulus",
    feature = "rsa-pss",
    feature = "p256-public-key",
    feature = "p256-signature"
))]
pub mod search_batches;

#[cfg(all(
    test,
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

pub mod search;
