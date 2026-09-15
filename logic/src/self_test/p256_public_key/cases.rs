use super::Case;

pub const CASES: &[Case] = &[
    Case {
        slot: 118,
        label: "p256 public key hmac derivation",
        kernel: "kernel_self_test_p256_public_key",
        gpu_skip: None,
    },
    Case {
        slot: 119,
        label: "p256 public key scalar derivation",
        kernel: "kernel_self_test_p256_public_key",
        gpu_skip: None,
    },
    Case {
        slot: 120,
        label: "p256 public key generator",
        kernel: "kernel_self_test_p256_public_key",
        gpu_skip: None,
    },
    Case {
        slot: 121,
        label: "p256 public key point double",
        kernel: "kernel_self_test_p256_public_key",
        gpu_skip: None,
    },
    Case {
        slot: 122,
        label: "p256 public key zero scalar rejected",
        kernel: "kernel_self_test_p256_public_key",
        gpu_skip: None,
    },
    Case {
        slot: 123,
        label: "p256 public key order scalar rejected",
        kernel: "kernel_self_test_p256_public_key",
        gpu_skip: None,
    },
    Case {
        slot: 124,
        label: "p256 public key x encoding",
        kernel: "kernel_self_test_p256_public_key",
        gpu_skip: None,
    },
    Case {
        slot: 125,
        label: "p256 public key y encoding",
        kernel: "kernel_self_test_p256_public_key",
        gpu_skip: None,
    },
    Case {
        slot: 153,
        label: "end-to-end p256 public candidate pipeline",
        kernel: "kernel_self_test_p256_public_key",
        gpu_skip: None,
    },
];
