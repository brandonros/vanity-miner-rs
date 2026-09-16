include!("../../../common/self_test_slots.rs");

#[test]
fn writes_every_owned_slot_without_touching_guards() {
    check_slots(
        "kernel_self_test_ethereum",
        kernel_self_test_ethereum_device::kernel_self_test_ethereum,
    );
}
