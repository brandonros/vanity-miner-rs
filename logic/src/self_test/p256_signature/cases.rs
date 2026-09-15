use super::Case;

pub const CASES: &[Case] = &[
    Case {
        slot: 126,
        label: "p256 signature rfc6979 sample",
        kernel: "kernel_self_test_p256_signature",
        gpu_skip: None,
    },
    Case {
        slot: 127,
        label: "p256 signature rfc6979 test",
        kernel: "kernel_self_test_p256_signature",
        gpu_skip: None,
    },
    Case {
        slot: 128,
        label: "p256 signature ephemeral r",
        kernel: "kernel_self_test_p256_signature",
        gpu_skip: None,
    },
    Case {
        slot: 129,
        label: "p256 signature ephemeral signature",
        kernel: "kernel_self_test_p256_signature",
        gpu_skip: None,
    },
    Case {
        slot: 130,
        label: "p256 signature zero nonce rejected",
        kernel: "kernel_self_test_p256_signature",
        gpu_skip: None,
    },
    Case {
        slot: 131,
        label: "p256 signature low s",
        kernel: "kernel_self_test_p256_signature",
        gpu_skip: None,
    },
    Case {
        slot: 132,
        label: "p256 signature high s",
        kernel: "kernel_self_test_p256_signature",
        gpu_skip: None,
    },
    Case {
        slot: 133,
        label: "p256 signature message window carry",
        kernel: "kernel_self_test_p256_signature",
        gpu_skip: None,
    },
    Case {
        slot: 134,
        label: "p256 signature ephemeral hmac",
        kernel: "kernel_self_test_p256_signature",
        gpu_skip: None,
    },
    Case {
        slot: 154,
        label: "end-to-end p256 signature candidate pipeline",
        kernel: "kernel_self_test_p256_signature",
        gpu_skip: None,
    },
];
