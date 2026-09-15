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
        let matches = self.matches_found.load(Ordering::Relaxed);
        println!("{}", self.format_progress(tested, matches, elapsed));
    }

    fn format_progress(&self, tested: u64, matches: usize, elapsed: Duration) -> String {
        let unit = *self.unit.lock().unwrap_or_else(|e| e.into_inner());
        let seconds = elapsed.as_secs_f64().max(1e-9);
        let rate = tested as f64 / seconds;
        let per_match = if matches == 0 {
            "n/a".to_string()
        } else {
            format!("{:.2}", tested as f64 / matches as f64)
        };
        let seconds_per_match = if matches == 0 {
            "n/a".to_string()
        } else {
            format!("{:.6}", seconds / matches as f64)
        };
        format!(
            "GLOBAL STATS: {} prefix, {} suffix | {matches} matches in {seconds:.2}s\n  {rate:.2} {unit}/sec ({:.2}/sec average per device/worker) | {:.6} matches/sec ({seconds_per_match}s/match)\n  {tested} total {unit} | {per_match} {unit}/match",
            self.vanity_prefix_length,
            self.vanity_suffix_length,
            rate / self.num_devices.max(1) as f64,
            matches as f64 / seconds,
        )
    }

    pub fn add_matches(&self, matches: usize) {
        self.matches_found.fetch_add(matches, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unified_report_uses_real_units_and_defined_match_rates() {
        let stats = GlobalStats::new(2, 4, 1);
        for unit in [
            "candidates",
            "q candidates",
            "keys",
            "salts",
            "messages",
            "nonces",
        ] {
            stats.set_unit(unit);
            let report = stats.format_progress(1000, 5, Duration::from_secs(2));
            assert!(report.contains("4 prefix, 1 suffix | 5 matches in 2.00s"));
            assert!(report.contains(&format!("500.00 {unit}/sec")));
            assert!(report.contains("250.00/sec average per device/worker"));
            assert!(report.contains("2.500000 matches/sec"));
            assert!(report.contains(&format!("1000 total {unit} | 200.00 {unit}/match")));
            assert!(report.contains("0.400000s/match"));
        }
        let report = stats.format_progress(0, 0, Duration::ZERO);
        assert!(report.contains("n/a"));
        assert!(!report.contains("NaN"));
        assert!(!report.contains("inf"));
    }

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
