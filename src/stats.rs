use std::sync::atomic::Ordering;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

pub struct GlobalStats {
    num_devices: usize,
    vanity_prefix_length: usize,
    vanity_suffix_length: usize,
    matches_found: AtomicUsize,
    total_operations: AtomicU64,
    start_time: AtomicU64,
}

impl GlobalStats {
    pub fn new(num_devices: usize, vanity_prefix_length: usize, vanity_suffix_length: usize) -> Self {
        let now_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Self {
            num_devices,
            vanity_prefix_length,
            vanity_suffix_length,
            matches_found: AtomicUsize::new(0),
            total_operations: AtomicU64::new(0),
            start_time: AtomicU64::new(now_nanos),
        }
    }

    pub fn add_launch(&self, operations: usize) {
        self.total_operations.fetch_add(operations as u64, Ordering::Relaxed);
    }

    pub fn add_matches(&self, matches: usize) {
        self.matches_found.fetch_add(matches, Ordering::Relaxed);
    }

    pub fn print_stats(&self, device_id: usize, matches_this_launch: f32) {
        let vanity_prefix_length = self.vanity_prefix_length;
        let vanity_suffix_length = self.vanity_suffix_length;
        let matches_found = self.matches_found.load(Ordering::Relaxed);
        let total_operations = self.total_operations.load(Ordering::Relaxed);
        let start_time = self.start_time.load(Ordering::Relaxed);

        let current_time_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let elapsed_nanos = current_time_nanos - start_time;
        let elapsed = Duration::from_nanos(elapsed_nanos);
        let elapsed_seconds = elapsed.as_secs_f64();
        let operations_per_second = total_operations as f64 / elapsed_seconds / 1_000_000.0;
        let device_operations_per_second = total_operations as f64 / elapsed_seconds / self.num_devices as f64 / 1_000_000.0;
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
        println!("[{device_id}] GLOBAL STATS ({vanity_prefix_length} prefix length, {vanity_suffix_length} suffix length): Found {matches_found} matches in {elapsed_seconds:.2}s ({match_eta:.6}s/match, {matches_per_second:.6} matches/sec, {operations_per_second:.2}M ops/sec, {device_operations_per_second:.4}M ops/sec/device, {operations_per_match:.4}M ops/match, {formatted_total_operations:.2}M ops)");
    }
}
