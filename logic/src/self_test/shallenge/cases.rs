use super::Case;

pub const CASES: &[Case] = &[
    Case {
        slot: 9,
        label: "sha256 variable",
        kernel: "kernel_self_test_shallenge",
        gpu_skip: None,
    },
    Case {
        slot: 25,
        label: "shallenge hash",
        kernel: "kernel_self_test_shallenge",
        gpu_skip: None,
    },
    Case {
        slot: 26,
        label: "shallenge nonce_len",
        kernel: "kernel_self_test_shallenge",
        gpu_skip: None,
    },
    Case {
        slot: 27,
        label: "shallenge is_better",
        kernel: "kernel_self_test_shallenge",
        gpu_skip: None,
    },
    Case {
        slot: 28,
        label: "compare_hashes lt",
        kernel: "kernel_self_test_shallenge",
        gpu_skip: None,
    },
    Case {
        slot: 29,
        label: "compare_hashes gt",
        kernel: "kernel_self_test_shallenge",
        gpu_skip: None,
    },
    Case {
        slot: 30,
        label: "compare_hashes eq",
        kernel: "kernel_self_test_shallenge",
        gpu_skip: None,
    },
    Case {
        slot: 44,
        label: "xoroshiro base64 nonce",
        kernel: "kernel_self_test_shallenge",
        gpu_skip: None,
    },
];
