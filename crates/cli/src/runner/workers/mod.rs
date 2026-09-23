//! Join every scoped worker, preserving the first failure and a verified winner.

#[cfg(feature = "crypto-cli")]
pub(crate) fn join<'scope, T>(
    handles: Vec<std::thread::ScopedJoinHandle<'scope, Result<Option<T>, String>>>,
    control: &crate::runner::session::SearchControl,
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

#[cfg(all(
    not(feature = "gpu"),
    any(
        feature = "solana",
        feature = "bitcoin",
        feature = "ethereum",
        feature = "shallenge"
    )
))]
pub(crate) mod cpu;
pub mod device;
