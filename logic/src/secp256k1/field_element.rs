// Field element arithmetic implementation for secp256k1
// Working with 256-bit integers in little-endian format for easier arithmetic

use crate::error::Error;
use crate::constants;

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
    
    // 512-bit modular reduction using proper division algorithm
    fn reduce_512_bit_mod_prime(&self, value: &[u64; 8]) -> FieldElement {
        println!("DEBUG: reduce_512_bit_mod_prime called");
        println!("DEBUG: input = {:016x} {:016x} {:016x} {:016x} {:016x} {:016x} {:016x} {:016x}", 
                 value[7], value[6], value[5], value[4], value[3], value[2], value[1], value[0]);
        
        // For secp256k1, we can use the special form of the prime for fast reduction
        // p = 2^256 - 2^32 - 977, so 2^256 ≡ 2^32 + 977 (mod p)
        // The offset value is 0x1000003D1 = 2^32 + 977
        
        // Split the 512-bit number into low and high 256-bit parts
        let low = [value[0], value[1], value[2], value[3]];   // Low 256 bits
        let high = [value[4], value[5], value[6], value[7]];  // High 256 bits
        
        println!("DEBUG: low  = {:016x} {:016x} {:016x} {:016x}", low[3], low[2], low[1], low[0]);
        println!("DEBUG: high = {:016x} {:016x} {:016x} {:016x}", high[3], high[2], high[1], high[0]);
        
        // If high part is zero, just reduce the low part
        if Self::is_zero_u64(&high) {
            println!("DEBUG: high part is zero, just reducing low part");
            let mut result = low;
            Self::reduce_mod_prime(&mut result);
            return Self::from_u64_array(result);
        }
        
        // Convert high part to little-endian bytes and multiply by the offset
        // offset = 2^32 + 977 = 0x1000003D1
        let offset = 0x1000003D1u64;
        
        // We need to compute: low + high * offset (mod p)
        // Since high can be up to 256 bits and offset is 64 bits, 
        // the product can be up to 320 bits
        
        let mut result = [0u64; 6]; // 384 bits to be safe
        
        // Copy low part
        result[0] = low[0];
        result[1] = low[1];
        result[2] = low[2];
        result[3] = low[3];
        
        // Add high * offset
        let mut carry = 0u128;
        for i in 0..4 {
            let prod = (high[i] as u128) * (offset as u128) + (result[i] as u128) + carry;
            result[i] = prod as u64;
            carry = prod >> 64;
        }
        
        // Handle remaining carry
        if carry > 0 {
            result[4] = carry as u64;
            carry >>= 64;
            if carry > 0 {
                result[5] = carry as u64;
            }
        }
        
        println!("DEBUG: after adding high*offset: {:016x} {:016x} {:016x} {:016x} {:016x} {:016x}", 
                 result[5], result[4], result[3], result[2], result[1], result[0]);
        
        // Now we need to reduce this potentially larger number mod p
        // If result fits in 256 bits, we're done
        if result[4] == 0 && result[5] == 0 {
            let mut final_result = [result[0], result[1], result[2], result[3]];
            Self::reduce_mod_prime(&mut final_result);
            return Self::from_u64_array(final_result);
        }
        
        // If result is larger, we need to reduce again
        // The high part now represents multiples of 2^256
        let high_part = [result[4], result[5], 0, 0];
        let low_part = [result[0], result[1], result[2], result[3]];
        
        println!("DEBUG: need second reduction, high={:016x} {:016x}, low={:016x} {:016x} {:016x} {:016x}", 
                 high_part[1], high_part[0], low_part[3], low_part[2], low_part[1], low_part[0]);
        
        // Recursively reduce: low + high * offset (mod p)
        let mut final_result = low_part;
        
        let mut carry = 0u128;
        for i in 0..2 { // only process non-zero high parts
            let prod = (high_part[i] as u128) * (offset as u128) + (final_result[i] as u128) + carry;
            final_result[i] = prod as u64;
            carry = prod >> 64;
        }
        
        // Propagate carry
        for i in 2..4 {
            if carry == 0 { break; }
            let sum = (final_result[i] as u128) + carry;
            final_result[i] = sum as u64;
            carry = sum >> 64;
        }
        
        // If there's still carry, it's small enough to ignore for this reduction level
        
        Self::reduce_mod_prime(&mut final_result);
        
        println!("DEBUG: final result = {:016x} {:016x} {:016x} {:016x}", 
                 final_result[3], final_result[2], final_result[1], final_result[0]);
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
        
        // Use Fermat's little theorem: a^(p-2) ≡ a^(-1) (mod p)
        // Use the precomputed p-2 constant
        let exp_bytes = constants::FIELD_PRIME_MINUS_2;
        
        let mut result = Self::one();
        let mut base = *self;
        
        // Binary exponentiation (little-endian bit order)
        for (byte_idx, &byte) in exp_bytes.iter().enumerate() {
            for bit in 0..8 {
                if (byte >> bit) & 1 == 1 {
                    result = result.mul(&base);
                }
                
                // Don't square on the very last bit to avoid unnecessary computation
                if byte_idx < 31 || bit < 7 {
                    base = base.square();
                }
            }
        }
        
        println!("DEBUG: invert: exponentiation complete, verifying result");
        
        // Verify the result: result * self should equal 1
        let check = result.mul(self);
        let one = Self::one();
        
        println!("DEBUG: invert: check = {:02x?}", check.data);
        println!("DEBUG: invert: one = {:02x?}", one.data);
        
        if check == one {
            println!("DEBUG: invert: verification successful");
            Ok(result)
        } else {
            println!("DEBUG: invert: verification failed");
            Err(Error::ArithmeticError)
        }
    }
    
    pub fn sqrt(&self) -> Result<Self, Error> {
        println!("DEBUG: sqrt called");
        // For secp256k1 prime p ≡ 3 (mod 4), we can use a^((p+1)/4) mod p
        
        // (p+1)/4 for secp256k1
        let exp_bytes = constants::FIELD_PRIME_PLUS_1_DIV_4;
        
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