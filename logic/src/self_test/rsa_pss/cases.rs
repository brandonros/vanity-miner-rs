use super::Case;

pub const CASES: &[Case] = &[
    Case {
        slot: 135,
        label: "rsa pss sha256",
        kernel: "kernel_self_test_rsa_pss",
        gpu_skip: None,
    },
    Case {
        slot: 136,
        label: "rsa pss mgf1 partial block",
        kernel: "kernel_self_test_rsa_pss",
        gpu_skip: None,
    },
    Case {
        slot: 137,
        label: "rsa pss salt32 encoding",
        kernel: "kernel_self_test_rsa_pss",
        gpu_skip: None,
    },
    Case {
        slot: 138,
        label: "rsa pss empty salt encoding",
        kernel: "kernel_self_test_rsa_pss",
        gpu_skip: None,
    },
    Case {
        slot: 139,
        label: "rsa pss maximum salt encoding",
        kernel: "kernel_self_test_rsa_pss",
        gpu_skip: None,
    },
    Case {
        slot: 140,
        label: "rsa pss oversized salt rejected",
        kernel: "kernel_self_test_rsa_pss",
        gpu_skip: None,
    },
    Case {
        slot: 141,
        label: "rsa pss salt carry",
        kernel: "kernel_self_test_rsa_pss",
        gpu_skip: None,
    },
    Case {
        slot: 142,
        label: "rsa pss crt known answer",
        kernel: "kernel_self_test_rsa_pss",
        gpu_skip: None,
    },
    Case {
        slot: 143,
        label: "rsa pss crt fault rejected",
        kernel: "kernel_self_test_rsa_pss",
        gpu_skip: None,
    },
    Case {
        slot: 144,
        label: "rsa pss crt modulus rejected",
        kernel: "kernel_self_test_rsa_pss",
        gpu_skip: None,
    },
    Case {
        slot: 155,
        label: "end-to-end rsa pss candidate pipeline",
        kernel: "kernel_self_test_rsa_pss",
        gpu_skip: Some(
            "temporarily disabled: RSA-PSS end-to-end GPU compilation takes ~7 min / 7.1 GiB and can OOM",
        ),
    },
];
