//! Join every scoped worker, preserving the first failure and a verified winner.
use crate::search_control::SearchControl;
use std::thread::ScopedJoinHandle;

pub(super) fn join<'scope, T>(
    handles: Vec<ScopedJoinHandle<'scope, Result<Option<T>, String>>>,
    control: &SearchControl,
    panic_message: &str,
) -> Result<Option<T>, String> {
    let mut winner = None;
    let mut error = None;
    for handle in handles {
        match handle.join() {
            Ok(Ok(Some(found))) => {
                if winner.is_none() {
                    winner = Some(found);
                }
            }
            Ok(Ok(None)) => {}
            Ok(Err(message)) => {
                control.cancel();
                error.get_or_insert(message);
            }
            Err(_) => {
                control.cancel();
                error.get_or_insert_with(|| panic_message.into());
            }
        }
    }
    error.map_or(Ok(winner), Err)
}
