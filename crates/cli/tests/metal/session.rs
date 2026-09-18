//! Lib-linked GPU test: accesses the real private session implementation.
use super::*;
use crate::runner::metal::transport::ShallengeTransport;

#[test]
#[ignore = "requires a built Metal kernel bundle and Apple GPU"]
fn real_search_updates_target_advances_counters_and_honors_cancellation() {
    let path = std::env::var_os("VANITY_METAL_ARTIFACTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("VANITY_METAL_BUNDLES")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/metal")
                })
                .join("shallenge")
        });
    let mut engine = ShallengeTransport::load(&path, 64, 32, true).unwrap();
    let best = Arc::new(RwLock::new(SharedBestHash::new([255; 32])));
    let control = SearchControl::new();
    control.set_batch_size(64).unwrap();
    let mut next = 0;
    for _ in 0..3 {
        control.set_device_launch_limit(Some(8));
        let previous = best.read().unwrap().get_current();
        let result = search(
            "brandonros",
            best.clone(),
            Some(12345),
            &control,
            |seed, target, user, start, count| {
                assert_eq!(start, next);
                assert_eq!(*target, previous);
                next += u64::from(count);
                engine.evaluate(seed, target, user, start, count)
            },
        )
        .unwrap();
        let current = best.read().unwrap().get_current();
        assert!(current <= previous);
        if result.is_some() {
            assert!(current < previous);
            assert!(control.resume_after_match());
        } else {
            break;
        }
    }
    assert!(next >= 64);
    control.interrupt();
    assert!(
        search(
            "brandonros",
            best,
            Some(12345),
            &control,
            |_, _, _, _, _| panic!("cancelled search dispatched")
        )
        .unwrap()
        .is_none()
    );
}
