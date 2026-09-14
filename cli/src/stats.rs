use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

pub struct GlobalStats {
    num_devices: usize,
    vanity_prefix_length: usize,
    vanity_suffix_length: usize,
    matches_found: AtomicUsize,
    total_operations: AtomicU64,
    start_time: Instant,
    unit: std::sync::Mutex<&'static str>,
}

impl GlobalStats {
    pub fn new(
        num_devices: usize,
        vanity_prefix_length: usize,
        vanity_suffix_length: usize,
    ) -> Self {
        Self {
            num_devices,
            vanity_prefix_length,
            vanity_suffix_length,
            matches_found: AtomicUsize::new(0),
            total_operations: AtomicU64::new(0),
            start_time: Instant::now(),
            unit: std::sync::Mutex::new("candidates"),
        }
    }

    pub fn add_launch(&self, operations: usize) {
        self.add_operations(operations as u64);
    }

    pub fn add_operations(&self, count: u64) {
        let _ = self
            .total_operations
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                Some(old.saturating_add(count))
            });
    }
    pub fn statistics(&self) -> (u64, Duration) {
        (
            self.total_operations.load(Ordering::Relaxed),
            self.start_time.elapsed(),
        )
    }
    pub fn set_unit(&self, unit: &'static str) {
        *self.unit.lock().unwrap_or_else(|e| e.into_inner()) = unit;
    }
    pub fn print_progress(&self) {
        let (tested, elapsed) = self.statistics();
        let unit = *self.unit.lock().unwrap_or_else(|e| e.into_inner());
        let seconds = elapsed.as_secs_f64().max(1e-9);
        println!(
            "{tested} {unit} in {seconds:.2}s ({:.2}/s)",
            tested as f64 / seconds.max(1e-9)
        );
    }

    pub fn add_matches(&self, matches: usize) {
        self.matches_found.fetch_add(matches, Ordering::Relaxed);
    }

    pub fn print_stats(&self, device_id: usize, matches_this_launch: u32) {
        let vanity_prefix_length = self.vanity_prefix_length;
        let vanity_suffix_length = self.vanity_suffix_length;
        let matches_found = self.matches_found.load(Ordering::Relaxed);
        let total_operations = self.total_operations.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64().max(1e-9);
        let operations_per_second = total_operations as f64 / elapsed_seconds / 1_000_000.0;
        let device_operations_per_second = total_operations as f64
            / elapsed_seconds
            / self.num_devices.max(1) as f64
            / 1_000_000.0;
        let matches_per_second = matches_found as f64 / elapsed_seconds;
        let match_eta = if matches_per_second > 0.0 {
            1.0 / matches_per_second
        } else {
            0.0
        };
        let operations_per_match = if matches_found > 0 {
            total_operations as f64 / matches_found as f64 / 1_000_000.0
        } else {
            0.0
        };
        let formatted_total_operations = total_operations as f64 / 1_000_000.0;

        println!("[{device_id}] Found {matches_this_launch} matches this launch");
        println!(
            "[{}] GLOBAL STATS: {} prefix, {} suffix | {} matches in {:.2}s",
            device_id, vanity_prefix_length, vanity_suffix_length, matches_found, elapsed_seconds
        );
        println!(
            "[{}]   {:.2}M ops/sec ({:.4}M/device) | {:.6} matches/sec ({:.6}s/match)",
            device_id,
            operations_per_second,
            device_operations_per_second,
            matches_per_second,
            match_eta
        );
        println!(
            "[{}]   {:.2}M total ops | {:.4}M ops/match",
            device_id, formatted_total_operations, operations_per_match
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_and_legacy_searches_share_the_same_counters() {
        let stats = std::sync::Arc::new(GlobalStats::new(2, 0, 0));
        let control = crate::search_control::SearchControl::with_stats(stats.clone());
        stats.add_launch(7);
        control.add_tested(11);
        assert_eq!(stats.statistics().0, 18);
        assert_eq!(control.statistics().0, 18);
        control.cancel();
        assert_eq!(stats.statistics().0, 18);
    }
    #[test]
    fn counters_saturate_instead_of_wrapping() {
        let stats = GlobalStats::new(1, 0, 0);
        stats.add_operations(u64::MAX);
        stats.add_operations(1);
        assert_eq!(stats.statistics().0, u64::MAX);
    }
}
