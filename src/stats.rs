use std::sync::atomic::Ordering;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;

pub struct GlobalStats {
    num_devices: usize,
    matches_found: AtomicUsize,
    total_operations: AtomicU64,
    start_time: Instant,
}

impl GlobalStats {
    pub fn new(num_devices: usize) -> Self {
        Self {
            num_devices,
            matches_found: AtomicUsize::new(0),
            total_operations: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    pub fn add_launch(&self, operations: usize) {
        self.total_operations.fetch_add(operations as u64, Ordering::Relaxed);
    }

    pub fn add_matches(&self, matches: usize) {
        self.matches_found.fetch_add(matches, Ordering::Relaxed);
    }

    pub fn print_stats(&self, device_id: usize, matches_this_launch: f32) {
        let matches = self.matches_found.load(Ordering::Relaxed);
        let operations = self.total_operations.load(Ordering::Relaxed);
        
        let elapsed = self.start_time.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let operations_per_second = operations as f64 / elapsed_seconds / 1_000_000.0;
        let device_operations_per_second = operations as f64 / elapsed_seconds / self.num_devices as f64 / 1_000_000.0;
        let matches_per_second = matches as f64 / elapsed_seconds;
        let match_eta = if matches_per_second > 0.0 {
            1.0 / matches_per_second
        } else {
            0.0
        };
        let operations_per_match = if matches > 0 {
            operations as f64 / matches as f64 / 1_000_000.0
        } else {
            0.0
        };

        println!("[{device_id}] Found {matches_this_launch} matches this launch");
        println!("[{device_id}] GLOBAL STATS: Found {matches} matches in {elapsed_seconds:.2}s ({match_eta:.6}s/match, {matches_per_second:.6} matches/sec, {operations_per_second:.2}M ops/sec, {device_operations_per_second:.2}M ops/sec/device, {operations_per_match:.0}M ops/match)");
    }
}