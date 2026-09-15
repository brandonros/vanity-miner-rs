use super::Case;

pub const CASES: &[Case] = &[
    Case {
        slot: 5,
        label: "secp256k1 uncompressed",
        kernel: "kernel_self_test_ethereum",
        gpu_skip: None,
    },
    Case {
        slot: 6,
        label: "keccak256 64bytes",
        kernel: "kernel_self_test_ethereum",
        gpu_skip: None,
    },
    Case {
        slot: 13,
        label: "ethereum priv",
        kernel: "kernel_self_test_ethereum",
        gpu_skip: None,
    },
    Case {
        slot: 14,
        label: "ethereum pub",
        kernel: "kernel_self_test_ethereum",
        gpu_skip: None,
    },
    Case {
        slot: 15,
        label: "ethereum address",
        kernel: "kernel_self_test_ethereum",
        gpu_skip: None,
    },
];
