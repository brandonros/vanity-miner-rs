include!("../../../common/self_test_slots.rs");

#[test]
fn writes_every_owned_slot_without_touching_guards() {
    check_slots(
        "kernel_self_test_p256_signature",
        kernel_self_test_p256_signature_device::kernel_self_test_p256_signature,
    );
}
