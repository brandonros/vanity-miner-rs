// Field element arithmetic implementation for secp256k1
// Working with 256-bit integers in little-endian format for easier arithmetic

use crate::error::Error;

// Field element operations (mod p)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldElement {
    pub data: [u8; 32],
}

impl FieldElement {
    pub fn new(data: [u8; 32]) -> Result<Self, Error> {
        // TODO: Validate data < field prime
        Ok(Self { data })
    }
    
    // Check if a little-endian byte array represents a small number (< 2^16)
    fn is_small_number_le(bytes: &[u8; 32]) -> bool {
        // Check if all bytes except the first two are zero (little-endian)
        for i in 2..32 {
            if bytes[i] != 0 {
                return false;
            }
        }
        true
    }
    
    // Convert little-endian bytes to small integer (assuming it fits in u16)
    fn bytes_to_small_int_le(bytes: &[u8; 32]) -> u16 {
        (bytes[0] as u16) | ((bytes[1] as u16) << 8)
    }
    
    pub fn zero() -> Self {
        Self { data: [0u8; 32] }
    }
    
    pub fn one() -> Self {
        let mut data = [0u8; 32];
        data[0] = 1; // Fixed: little-endian means LSB first
        Self { data }
    }

    // Convert to little-endian u64 array for easier arithmetic
    fn to_u64_array(&self) -> [u64; 4] {
        let mut result = [0u64; 4];
        for i in 0..4 {
            result[i] = u64::from_le_bytes([
                self.data[i * 8],
                self.data[i * 8 + 1],
                self.data[i * 8 + 2],
                self.data[i * 8 + 3],
                self.data[i * 8 + 4],
                self.data[i * 8 + 5],
                self.data[i * 8 + 6],
                self.data[i * 8 + 7],
            ]);
        }
        result
    }
    
    // Convert from little-endian u64 array back to bytes
    fn from_u64_array(array: [u64; 4]) -> Self {
        let mut data = [0u8; 32];
        for i in 0..4 {
            let bytes = array[i].to_le_bytes();
            data[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
        }
        Self { data }
    }
    
    // Get field prime as u64 array (little-endian)
    fn field_prime() -> [u64; 4] {
        // secp256k1 prime: 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
        // In little-endian u64 format:
        [
            0xFFFFFFFEFFFFFC2F, // low 64 bits
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, // high 64 bits
        ]
    }
    
    // Check if number is >= field prime
    fn is_ge_prime(a: &[u64; 4]) -> bool {
        let prime = Self::field_prime();
        // Compare from high to low for big-endian comparison
        for i in (0..4).rev() {
            match a[i].cmp(&prime[i]) {
                core::cmp::Ordering::Greater => {
                    println!("DEBUG: is_ge_prime returning true (greater at index {})", i);
                    return true;
                }
                core::cmp::Ordering::Less => {
                    println!("DEBUG: is_ge_prime returning false (less at index {})", i);
                    return false;
                }
                core::cmp::Ordering::Equal => continue,
            }
        }
        println!("DEBUG: is_ge_prime returning true (equal)");
        true // Equal to prime
    }
    
    // Subtract field prime (assumes a >= prime)
    fn sub_prime(a: &mut [u64; 4]) {
        let prime = Self::field_prime();
        let mut borrow = 0u64;
        
        println!("DEBUG: sub_prime before: {:016x} {:016x} {:016x} {:016x}", a[3], a[2], a[1], a[0]);
        
        for i in 0..4 {
            let (diff, b1) = a[i].overflowing_sub(prime[i]);
            let (final_diff, b2) = diff.overflowing_sub(borrow);
            a[i] = final_diff;
            borrow = (b1 as u64) + (b2 as u64);
        }
        
        println!("DEBUG: sub_prime after: {:016x} {:016x} {:016x} {:016x}", a[3], a[2], a[1], a[0]);
    }
    
    // Reduce modulo field prime with loop counter to prevent infinite loops
    fn reduce_mod_prime(a: &mut [u64; 4]) {
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 10; // Should never need more than 1-2 iterations
        
        println!("DEBUG: reduce_mod_prime input: {:016x} {:016x} {:016x} {:016x}", a[3], a[2], a[1], a[0]);
        
        while Self::is_ge_prime(a) {
            iterations += 1;
            if iterations > MAX_ITERATIONS {
                panic!("reduce_mod_prime: infinite loop detected after {} iterations", MAX_ITERATIONS);
            }
            println!("DEBUG: reduce_mod_prime iteration {}", iterations);
            Self::sub_prime(a);
        }
        
        println!("DEBUG: reduce_mod_prime output: {:016x} {:016x} {:016x} {:016x} (after {} iterations)", 
                 a[3], a[2], a[1], a[0], iterations);
    }
    
    pub fn add(&self, other: &Self) -> Self {
        println!("DEBUG: add called");
        let a = self.to_u64_array();
        let b = other.to_u64_array();
        let mut result = [0u64; 4];
        let mut carry = 0u64;
        
        // Add with carry
        for i in 0..4 {
            let (sum1, c1) = a[i].overflowing_add(b[i]);
            let (sum2, c2) = sum1.overflowing_add(carry);
            result[i] = sum2;
            carry = (c1 as u64) + (c2 as u64);
        }
        
        // Reduce modulo prime
        Self::reduce_mod_prime(&mut result);
        
        Self::from_u64_array(result)
    }
    
    pub fn sub(&self, other: &Self) -> Self {
        println!("DEBUG: sub called");
        let a = self.to_u64_array();
        let b = other.to_u64_array();
        let mut result = [0u64; 4];
        let mut borrow = 0u64;
        
        // Subtract with borrow
        for i in 0..4 {
            let (diff1, b1) = a[i].overflowing_sub(b[i]);
            let (diff2, b2) = diff1.overflowing_sub(borrow);
            result[i] = diff2;
            borrow = (b1 as u64) + (b2 as u64);
        }
        
        // If we borrowed, add the field prime
        if borrow != 0 {
            let prime = Self::field_prime();
            let mut carry = 0u64;
            for i in 0..4 {
                let (sum1, c1) = result[i].overflowing_add(prime[i]);
                let (sum2, c2) = sum1.overflowing_add(carry);
                result[i] = sum2;
                carry = (c1 as u64) + (c2 as u64);
            }
        }
        
        Self::from_u64_array(result)
    }

    pub fn square(&self) -> Self {
        // Square a field element mod p
        self.mul(self)
    }
    
    pub fn mul(&self, other: &Self) -> Self {
        println!("DEBUG: mul called");
        let a = self.to_u64_array();
        let b = other.to_u64_array();
        
        // 256-bit x 256-bit multiplication gives 512-bit result
        let mut result = [0u64; 8];
        
        // School-book multiplication
        for i in 0..4 {
            let mut carry = 0u128;
            for j in 0..4 {
                let prod = (a[i] as u128) * (b[j] as u128) + (result[i + j] as u128) + carry;
                result[i + j] = prod as u64;
                carry = prod >> 64;
            }
            if i + 4 < 8 {
                result[i + 4] = carry as u64;
            }
        }
        
        // Reduce 512-bit result modulo field prime
        self.reduce_512_bit_mod_prime(&result)
    }
    
    // Optimized 512-bit modular reduction - much faster version
    fn reduce_512_bit_mod_prime(&self, value: &[u64; 8]) -> FieldElement {
        // For secp256k1: p = 2^256 - 2^32 - 977
        // So 2^256 ≡ 2^32 + 977 (mod p)
        // We can reduce efficiently using this identity
        
        let low = [value[0], value[1], value[2], value[3]];
        let high = [value[4], value[5], value[6], value[7]];
        
        // If high part is zero, just reduce the low part
        if Self::is_zero_u64(&high) {
            let mut result = low;
            Self::reduce_mod_prime(&mut result);
            return Self::from_u64_array(result);
        }
        
        // Fast reduction using the secp256k1 prime structure
        // high * 2^256 ≡ high * (2^32 + 977) (mod p)
        
        // Convert to 128-bit arithmetic for precision
        let mut result = [0u128; 4];
        
        // Start with low part
        for i in 0..4 {
            result[i] = low[i] as u128;
        }
        
        // Add high * (2^32 + 977)
        let c = 977u128;
        let mut carry = 0u128;
        
        for i in 0..4 {
            if high[i] == 0 { continue; }
            
            let h = high[i] as u128;
            
            // Add high[i] * 977 to result[i]
            let prod = h * c + result[i] + carry;
            result[i] = prod & 0xFFFFFFFFFFFFFFFF;
            carry = prod >> 64;
            
            // Add high[i] * 2^32 to result[i+1] (if i < 3)
            if i < 3 {
                let shifted = h << 32;
                result[i + 1] += shifted;
            } else {
                // high[3] * 2^32 would overflow, so add high[3] * 977 instead
                carry += h * c;
            }
        }
        
        // Handle final carry
        if carry > 0 {
            // carry * 2^256 ≡ carry * (2^32 + 977) (mod p)
            result[0] += carry * c;
            if carry <= 0xFFFFFFFF {
                result[1] += carry << 32;
            } else {
                // Very large carry - add carry * 977 to low part
                result[0] += carry * c;
            }
        }
        
        // Convert back to u64 and handle any remaining carries
        let mut final_result = [0u64; 4];
        let mut carry = 0u64;
        
        for i in 0..4 {
            let val = result[i] + (carry as u128);
            final_result[i] = val as u64;
            carry = (val >> 64) as u64;
        }
        
        // Final reduction
        Self::reduce_mod_prime(&mut final_result);
        Self::from_u64_array(final_result)
    }
    
    fn is_zero_u64(a: &[u64; 4]) -> bool {
        a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0
    }
    
    pub fn negate(&self) -> Self {
        let zero = Self::zero();
        zero.sub(self)
    }
    
    pub fn invert(&self) -> Result<Self, Error> {
        println!("DEBUG: invert called on {:02x?}", self.data);
        
        // First check if input is zero
        if *self == Self::zero() {
            println!("DEBUG: invert: input is zero");
            return Err(Error::ArithmeticError);
        }
        
        // For small numbers, use precomputed values
        if Self::is_small_number_le(&self.data) {
            let small_val = Self::bytes_to_small_int_le(&self.data);
            if let Some(inv_bytes_be) = Self::invert_small_number(small_val) {
                let result = Self::from_bytes_be(&inv_bytes_be)?;
                
                // Verify the result
                let check = result.mul(self);
                if check == Self::one() {
                    println!("DEBUG: invert: small number inversion successful");
                    return Ok(result);
                }
            }
        }
        
        // For large numbers, use binary extended Euclidean algorithm
        println!("DEBUG: Using binary extended Euclidean algorithm for large number");
        
        // Binary Extended Euclidean Algorithm
        // We want to find x such that a*x ≡ 1 (mod p)
        
        let mut u = self.to_u64_array();
        let mut v = Self::field_prime();
        let mut x1 = [1u64, 0, 0, 0]; // coefficient of a
        let mut x2 = [0u64, 0, 0, 0]; // coefficient of p
        
        // Binary GCD with extended coefficients
        while !Self::is_zero_u64(&u) && !Self::is_zero_u64(&v) {
            // Make u even
            while Self::is_even_u64(&u) {
                Self::divide_by_2_u64(&mut u);
                if Self::is_even_u64(&x1) {
                    Self::divide_by_2_u64(&mut x1);
                } else {
                    Self::add_u64(&mut x1, &Self::field_prime());
                    Self::divide_by_2_u64(&mut x1);
                }
            }
            
            // Make v even
            while Self::is_even_u64(&v) {
                Self::divide_by_2_u64(&mut v);
                if Self::is_even_u64(&x2) {
                    Self::divide_by_2_u64(&mut x2);
                } else {
                    Self::add_u64(&mut x2, &Self::field_prime());
                    Self::divide_by_2_u64(&mut x2);
                }
            }
            
            // Subtract smaller from larger
            if Self::compare_u64(&u, &v) >= 0 {
                Self::subtract_u64(&mut u, &v);
                Self::subtract_mod_p(&mut x1, &x2);
            } else {
                Self::subtract_u64(&mut v, &u);
                Self::subtract_mod_p(&mut x2, &x1);
            }
        }
        
        // Result should be in x2 if v ended up as gcd
        if Self::is_one_u64(&v) {
            let result = Self::from_u64_array(x2);
            
            // Verify the result
            let check = result.mul(self);
            if check == Self::one() {
                println!("DEBUG: invert: binary EEA successful");
                Ok(result)
            } else {
                println!("DEBUG: invert: binary EEA verification failed");
                Err(Error::ArithmeticError)
            }
        } else {
            println!("DEBUG: invert: binary EEA failed - no inverse exists");
            Err(Error::ArithmeticError)
        }
    }
    
    // Helper functions for binary extended Euclidean algorithm
    
    fn is_even_u64(a: &[u64; 4]) -> bool {
        a[0] & 1 == 0
    }
    
    fn is_one_u64(a: &[u64; 4]) -> bool {
        a[0] == 1 && a[1] == 0 && a[2] == 0 && a[3] == 0
    }
    
    fn divide_by_2_u64(a: &mut [u64; 4]) {
        let mut carry = 0u64;
        for i in (0..4).rev() {
            let new_carry = (a[i] & 1) << 63;
            a[i] = (a[i] >> 1) | carry;
            carry = new_carry;
        }
    }
    
    fn add_u64(a: &mut [u64; 4], b: &[u64; 4]) {
        let mut carry = 0u64;
        for i in 0..4 {
            let (sum1, c1) = a[i].overflowing_add(b[i]);
            let (sum2, c2) = sum1.overflowing_add(carry);
            a[i] = sum2;
            carry = (c1 as u64) + (c2 as u64);
        }
    }
    
    fn subtract_u64(a: &mut [u64; 4], b: &[u64; 4]) {
        let mut borrow = 0u64;
        for i in 0..4 {
            let (diff1, b1) = a[i].overflowing_sub(b[i]);
            let (diff2, b2) = diff1.overflowing_sub(borrow);
            a[i] = diff2;
            borrow = (b1 as u64) + (b2 as u64);
        }
    }
    
    fn subtract_mod_p(a: &mut [u64; 4], b: &[u64; 4]) {
        // Compute a - b (mod p)
        let mut result = *a;
        
        // If a >= b, compute a - b
        if Self::compare_u64(a, b) >= 0 {
            Self::subtract_u64(&mut result, b);
        } else {
            // If a < b, compute a + p - b
            let prime = Self::field_prime();
            Self::add_u64(&mut result, &prime);
            Self::subtract_u64(&mut result, b);
        }
        
        *a = result;
    }
    
    fn compare_u64(a: &[u64; 4], b: &[u64; 4]) -> i32 {
        for i in (0..4).rev() {
            if a[i] > b[i] { return 1; }
            if a[i] < b[i] { return -1; }
        }
        0
    }
    
    // Convert big-endian bytes to FieldElement
    fn from_bytes_be(bytes: &[u8; 32]) -> Result<Self, Error> {
        let mut data = [0u8; 32];
        for i in 0..32 {
            data[i] = bytes[31 - i]; // Reverse for little-endian storage
        }
        Ok(Self { data })
    }
    
    // Invert small numbers using precomputed values or simple iteration
    fn invert_small_number(n: u16) -> Option<[u8; 32]> {
        match n {
            1 => {
                // 1^(-1) = 1
                let mut result = [0u8; 32];
                result[31] = 1;
                Some(result)
            }
            2 => {
                // 2^(-1) = (p+1)/2 for secp256k1
                let result = [
                    0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0x7F, 0xFF, 0xFE, 0x18,
                ];
                Some(result)
            }
            _ => None // For other small numbers, we'd need to compute or precompute
        }
    }
    
    pub fn sqrt(&self) -> Result<Self, Error> {
        println!("DEBUG: sqrt called");
        // For secp256k1 prime p ≡ 3 (mod 4), we can use a^((p+1)/4) mod p
        
        // (p+1)/4 for secp256k1
        let exp_bytes = [
            0x4C, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x3F,
        ];
        
        let mut result = Self::one();
        let mut base = *self;
        let mut bit_count = 0;
        const MAX_BITS: usize = 256 * 8; // Safety limit
        
        // Binary exponentiation
        for &byte in exp_bytes.iter() {
            for bit in 0..8 {
                bit_count += 1;
                if bit_count > MAX_BITS {
                    panic!("sqrt: too many bits processed");
                }
                
                if (byte >> bit) & 1 == 1 {
                    result = result.mul(&base);
                }
                base = base.square();
            }
        }
        
        // Verify that result^2 = self
        let check = result.square();
        if check == *self {
            Ok(result)
        } else {
            Err(Error::ArithmeticError)
        }
    }
}