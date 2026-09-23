//! Base58 probes owned by the bitcoin self-test kernel.
use super::*;

register_self_test! {
    /// base58 var-len leading-zero
    fn base58_var_len_leading_zero() -> u32 {
        let mut out = [0u8; 64];
        let n = base58_encode(&BASE58_LEADZERO_INPUT, &mut out);
        if n != BASE58_LEADZERO_EXPECTED.len() {
            return 0;
        }
        let mut i = 0;
        while i < n {
            if out[i] != BASE58_LEADZERO_EXPECTED[i] {
                return 0;
            }
            i += 1;
        }
        1
    }
}
