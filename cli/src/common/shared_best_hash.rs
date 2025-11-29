pub struct SharedBestHash {
    hash: [u8; 32],
}

impl SharedBestHash {
    pub fn new(initial_hash: [u8; 32]) -> Self {
        Self { hash: initial_hash }
    }

    pub fn update_if_better(&mut self, new_hash: [u8; 32]) -> bool {
        // Compare hashes lexicographically (smaller is better)
        if new_hash < self.hash {
            self.hash = new_hash;
            true
        } else {
            false
        }
    }

    pub fn get_current(&self) -> [u8; 32] {
        self.hash
    }
}
