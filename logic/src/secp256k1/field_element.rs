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
    
    pub fn zero() -> Self {
        Self { data: [0u8; 32] }
    }
    
    pub fn one() -> Self {
        let mut data = [0u8; 32];
        data[31] = 1;
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
    
    // Get field prime as u64 array
    fn field_prime() -> [u64; 4] {
        // secp256k1 prime: 2^256 - 2^32 - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1
        // = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
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
        for i in (0..4).rev() {
            match a[i].cmp(&prime[i]) {
                core::cmp::Ordering::Greater => return true,
                core::cmp::Ordering::Less => return false,
                core::cmp::Ordering::Equal => continue,
            }
        }
        true // Equal to prime
    }
    
    // Subtract field prime (assumes a >= prime)
    fn sub_prime(a: &mut [u64; 4]) {
        let prime = Self::field_prime();
        let mut borrow = 0u64;
        
        for i in 0..4 {
            let (diff, b1) = a[i].overflowing_sub(prime[i]);
            let (final_diff, b2) = diff.overflowing_sub(borrow);
            a[i] = final_diff;
            borrow = (b1 as u64) + (b2 as u64);
        }
    }
    
    // Reduce modulo field prime
    fn reduce_mod_prime(a: &mut [u64; 4]) {
        while Self::is_ge_prime(a) {
            Self::sub_prime(a);
        }
    }
    
    pub fn add(&self, other: &Self) -> Self {
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
            result[i + 4] = carry as u64;
        }
        
        // Reduce 512-bit result modulo field prime
        // This is a simplified reduction - for production use Barrett or Montgomery reduction
        self.reduce_512_bit_mod_prime(&result)
    }
    
    // Simplified 512-bit modular reduction
    fn reduce_512_bit_mod_prime(&self, value: &[u64; 8]) -> FieldElement {
        // For secp256k1, we can use the special form of the prime for faster reduction
        // This is a basic implementation - Barrett reduction would be more efficient
        
        // Convert to big integer representation and do division
        // For now, we'll use a simple approach with repeated subtraction
        let mut high = [value[4], value[5], value[6], value[7]];
        let mut low = [value[0], value[1], value[2], value[3]];
        
        // If high part is non-zero, we need to reduce
        while !Self::is_zero_u64(&high) || Self::is_ge_prime(&low) {
            // Subtract prime from low part
            if Self::is_ge_prime(&low) {
                Self::sub_prime(&mut low);
            } else if !Self::is_zero_u64(&high) {
                // Add 2^256 to low part (effectively subtracting from high)
                // and then subtract prime
                Self::add_2_256_mod_prime(&mut low, &mut high);
            }
        }
        
        Self::from_u64_array(low)
    }
    
    fn is_zero_u64(a: &[u64; 4]) -> bool {
        a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0
    }
    
    fn add_2_256_mod_prime(low: &mut [u64; 4], high: &mut [u64; 4]) {
        // Add 2^256 ≡ 2^32 + 2^9 + 2^8 + 2^7 + 2^6 + 2^4 + 1 (mod p)
        let addend = [
            0x1000003D1, // 2^32 + 2^9 + 2^8 + 2^7 + 2^6 + 2^4 + 1
            0,
            0,
            0,
        ];
        
        let mut carry = 0u64;
        for i in 0..4 {
            let (sum1, c1) = low[i].overflowing_add(addend[i]);
            let (sum2, c2) = sum1.overflowing_add(carry);
            low[i] = sum2;
            carry = (c1 as u64) + (c2 as u64);
        }
        
        // Subtract 1 from high part
        if high[0] > 0 {
            high[0] -= 1;
        } else {
            // Propagate borrow
            for i in 1..4 {
                if high[i] > 0 {
                    high[i] -= 1;
                    break;
                }
                high[i] = u64::MAX;
            }
        }
    }
    
    pub fn negate(&self) -> Self {
        let zero = Self::zero();
        zero.sub(self)
    }
    
    pub fn invert(&self) -> Result<Self, Error> {
        // Use Fermat's little theorem: a^(p-2) ≡ a^(-1) (mod p)
        // p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
        
        let mut result = *self;
        let exp = [
            0xFFFFFFFEFFFFFC2D, // p-2 in little-endian u64 format
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
        ];
        
        // Binary exponentiation
        let mut base = *self;
        result = Self::one();
        
        for word in exp.iter() {
            for bit in 0..64 {
                if (word >> bit) & 1 == 1 {
                    result = result.mul(&base);
                }
                base = base.mul(&base);
            }
        }
        
        // Check if result * self = 1
        let check = result.mul(self);
        if check == Self::one() {
            Ok(result)
        } else {
            Err(Error::ArithmeticError)
        }
    }
    
    pub fn sqrt(&self) -> Result<Self, Error> {
        // For secp256k1 prime p ≡ 3 (mod 4), we can use a^((p+1)/4) mod p
        // (p+1)/4 = 0x3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFF0C
        
        let exp = [
            0x3FFFFFFFBFFFFF0C, // (p+1)/4 in little-endian u64 format
            0x3FFFFFFFFFFFFFFF,
            0x3FFFFFFFFFFFFFFF,
            0x3FFFFFFFFFFFFFFF,
        ];
        
        let mut result = Self::one();
        let mut base = *self;
        
        // Binary exponentiation
        for word in exp.iter() {
            for bit in 0..64 {
                if (word >> bit) & 1 == 1 {
                    result = result.mul(&base);
                }
                base = base.mul(&base);
            }
        }
        
        // Verify that result^2 = self
        let check = result.mul(&result);
        if check == *self {
            Ok(result)
        } else {
            Err(Error::ArithmeticError)
        }
    }
}