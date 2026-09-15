use super::Case;

pub const CASES: &[Case] = &[
    Case {
        slot: 145,
        label: "rsa modulus multiplication carry",
        kernel: "kernel_self_test_rsa_modulus",
        gpu_skip: None,
    },
    Case {
        slot: 146,
        label: "rsa modulus progression carry",
        kernel: "kernel_self_test_rsa_modulus",
        gpu_skip: None,
    },
    Case {
        slot: 147,
        label: "rsa modulus prime filter",
        kernel: "kernel_self_test_rsa_modulus",
        gpu_skip: None,
    },
    Case {
        slot: 148,
        label: "rsa modulus pseudoprime rejected",
        kernel: "kernel_self_test_rsa_modulus",
        gpu_skip: None,
    },
    Case {
        slot: 149,
        label: "rsa modulus zero stride rejected",
        kernel: "kernel_self_test_rsa_modulus",
        gpu_skip: None,
    },
    Case {
        slot: 150,
        label: "rsa modulus upper bound rejected",
        kernel: "kernel_self_test_rsa_modulus",
        gpu_skip: None,
    },
    Case {
        slot: 151,
        label: "rsa modulus equal factors rejected",
        kernel: "kernel_self_test_rsa_modulus",
        gpu_skip: None,
    },
    Case {
        slot: 152,
        label: "rsa modulus undersized factor rejected",
        kernel: "kernel_self_test_rsa_modulus",
        gpu_skip: None,
    },
    Case {
        slot: 156,
        label: "end-to-end rsa modulus candidate pipeline",
        kernel: "kernel_self_test_rsa_modulus",
        gpu_skip: None,
    },
];
