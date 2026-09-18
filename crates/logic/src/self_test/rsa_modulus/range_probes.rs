//! Focused range probes for the rsa_modulus self-test.
use crate::modes::rsa_modulus::{self as pipeline, SearchConfig, Task};
use crate::self_test::black_box;
use crypto_bigint::{Encoding, U1024, U2048};

register_self_test! {
    /// rsa empty interval clears task
    fn range_empty() -> u32 {
        let mut config = super::device_config();
        let n = U2048::from_be_slice(&config.lower)
            .wrapping_add(&U2048::ONE)
            .to_be_bytes();
        // [p*q+1, p*q+1] contains no multiple of p.
        config.lower = n;
        config.upper = n;
        config.suffix_bits = 1;
        let mut task = initial_task();
        u32::from(pipeline::prepare_range(&black_box(config), &mut task) == Ok(false) && cleared(&task))
    }
}

register_self_test! {
    /// rsa prepared five-candidate progression
    fn range_multiple() -> u32 {
        let config = black_box(five_candidate_config());
        let mut task = initial_task();
        if pipeline::prepare_range(&config, &mut task) != Ok(true) {
            return 0;
        }
        if task.state != 2
            || task.winner != 0
            || task.first != super::SELF_TEST_RSA_Q
            || task.count != U1024::from_u32(5).to_be_bytes()
            || task.remaining != task.count
            || task.skip_count != [0; 128]
            || task.skip_start != [0; 128]
        {
            return 0;
        }
        // Independent Python HMAC-SHA256 oracle: seed=[42;32], worker=7, id=9,
        // domain rsa-range-start, attempt=0: masked sample is 0 for bound=5.
        if task.cursor != [0; 128] {
            return 0;
        }
        for offset in 0..5 {
            let mut expected = super::SELF_TEST_RSA_Q;
            expected[127] += 2 * offset as u8;
            if pipeline::q_at(&config, &task, black_box(offset)) != Some(expected) {
                return 0;
            }
        }
        u32::from(pipeline::q_at(&config, &task, black_box(5)).is_none())
    }
}

register_self_test! {
    /// rsa range exhausted without winner
    fn range_exhausted() -> u32 {
        let config = black_box(five_candidate_config());
        let mut task = initial_task();
        if pipeline::prepare_range(&config, &mut task) != Ok(true) {
            return 0;
        }
        if task.winner != 0 {
            return 0;
        }
        pipeline::finish_tile(&mut task, black_box(5));
        u32::from(cleared(&task) && pipeline::q_at(&config, &task, black_box(0)).is_none())
    }
}

register_self_test! {
    /// rsa partial final tile and zero assigned work
    fn range_partial_tile() -> u32 {
        let config = black_box(five_candidate_config());
        let mut task = initial_task();
        if pipeline::prepare_range(&config, &mut task) != Ok(true) {
            return 0;
        }
        pipeline::finish_tile(&mut task, black_box(0));
        if task.cursor != [0; 128] || task.remaining != U1024::from_u32(5).to_be_bytes() {
            return 0;
        }
        pipeline::finish_tile(&mut task, black_box(2));
        if task.state != 2
            || task.winner != 0
            || task.cursor != U1024::from_u32(2).to_be_bytes()
            || task.remaining != U1024::from_u32(3).to_be_bytes()
        {
            return 0;
        }
        for offset in 0..3 {
            let mut expected = super::SELF_TEST_RSA_Q;
            expected[127] += 4 + 2 * offset as u8;
            if pipeline::q_at(&config, &task, black_box(offset)) != Some(expected) {
                return 0;
            }
        }
        if pipeline::q_at(&config, &task, black_box(3)).is_some() {
            return 0;
        }
        pipeline::finish_tile(&mut task, black_box(4)); // A four-lane tile only has three candidates left.
        u32::from(cleared(&task))
    }
}

fn five_candidate_config() -> SearchConfig {
    let mut config = super::device_config();
    let p = U1024::from_be_slice(&super::SELF_TEST_RSA_P);
    let mut last = super::SELF_TEST_RSA_Q;
    last[127] += 8;
    config.upper = p.mul(&U1024::from_be_slice(&last)).to_be_bytes();
    config.suffix_bits = 1;
    config
}

fn initial_task() -> Task {
    black_box(Task {
        p: super::SELF_TEST_RSA_P,
        id: 9,
        state: 1,
        winner: 1,
        ..Task::EMPTY
    })
}

// Check every field explicitly; never inspect repr(C) padding.
fn cleared(task: &Task) -> bool {
    task.p == [0; 128]
        && task.first == [0; 128]
        && task.count == [0; 128]
        && task.cursor == [0; 128]
        && task.remaining == [0; 128]
        && task.skip_start == [0; 128]
        && task.skip_count == [0; 128]
        && task.id == 0
        && task.state == 0
        && task.winner == 0
}
