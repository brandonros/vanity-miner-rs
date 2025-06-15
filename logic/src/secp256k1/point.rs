use super::field_element::FieldElement;
use super::error::Error;
use super::constants;

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: FieldElement,
    pub y: FieldElement,
    pub infinity: bool,
}

impl Point {
    pub fn new(x: FieldElement, y: FieldElement) -> Result<Self, Error> {
        let point = Self { x, y, infinity: false };
        if point.is_on_curve() {
            Ok(point)
        } else {
            println!("DEBUG: Point not on curve: x={:02x?}, y={:02x?}", x.data, y.data);
            Err(Error::InvalidPublicKey)
        }
    }
    
    pub fn infinity() -> Self {
        Self {
            x: FieldElement::zero(),
            y: FieldElement::zero(),
            infinity: true,
        }
    }
    
    pub fn generator() -> Self {
        Self {
            x: FieldElement::new(constants::GENERATOR_X).unwrap(),
            y: FieldElement::new(constants::GENERATOR_Y).unwrap(),
            infinity: false,
        }
    }
    
    fn is_on_curve(&self) -> bool {
        if self.infinity {
            return true;
        }
        
        // Check if y^2 = x^3 + 7 (secp256k1 curve equation)
        let y_squared = self.y.square();
        let x_cubed = self.x.square().mul(&self.x);
        let seven = {
            let mut data = [0u8; 32];
            data[0] = 7; // Fixed: little-endian
            FieldElement::new(data).unwrap()
        };
        let right_side = x_cubed.add(&seven);
        
        let is_valid = y_squared == right_side;
        if !is_valid {
            println!("DEBUG: Curve check failed:");
            println!("  y^2 = {:02x?}", y_squared.data);
            println!("  x^3+7 = {:02x?}", right_side.data);
        }
        is_valid
    }
    
    // Point addition
    pub fn add(&self, other: &Self) -> Self {
        println!("DEBUG: Point::add called");
        
        if self.infinity { 
            println!("DEBUG: self is infinity, returning other");
            return *other; 
        }
        if other.infinity { 
            println!("DEBUG: other is infinity, returning self");
            return *self; 
        }
        
        if self.x == other.x {
            if self.y == other.y {
                println!("DEBUG: Same point, calling double");
                return self.double();
            } else {
                println!("DEBUG: Same x but different y, returning infinity");
                return Point::infinity();
            }
        }
        
        // λ = (y₂ - y₁) / (x₂ - x₁)
        let dx = other.x.sub(&self.x);
        let dy = other.y.sub(&self.y);
        
        println!("DEBUG: dx = {:02x?}", dx.data);
        println!("DEBUG: dy = {:02x?}", dy.data);
        
        // Check if dx is zero (which would make inversion fail)
        if dx == FieldElement::zero() {
            println!("DEBUG: dx is zero, this shouldn't happen in add");
            return Point::infinity();
        }
        
        let dx_inv = match dx.invert() {
            Ok(inv) => inv,
            Err(e) => {
                println!("DEBUG: Failed to invert dx: {:?}", e);
                return Point::infinity();
            }
        };
        
        let lambda = dy.mul(&dx_inv);
        
        // x₃ = λ² - x₁ - x₂
        let x3 = lambda.square().sub(&self.x).sub(&other.x);
        
        // y₃ = λ(x₁ - x₃) - y₁
        let y3 = lambda.mul(&self.x.sub(&x3)).sub(&self.y);
        
        Point { x: x3, y: y3, infinity: false }
    }
    
    // Point doubling
    pub fn double(&self) -> Self {
        println!("DEBUG: Point::double called");
        
        if self.infinity { 
            println!("DEBUG: infinity point, returning infinity");
            return *self; 
        }
        
        if self.y == FieldElement::zero() { 
            println!("DEBUG: y is zero, returning infinity");
            return Point::infinity(); 
        }
        
        // λ = 3x₁² / (2y₁)
        let x_squared = self.x.square();
        let three_x_squared = x_squared.add(&x_squared).add(&x_squared);
        let two_y = self.y.add(&self.y);
        
        println!("DEBUG: 3x² = {:02x?}", three_x_squared.data);
        println!("DEBUG: 2y = {:02x?}", two_y.data);
        
        // Check if 2y is zero (which would make inversion fail)
        if two_y == FieldElement::zero() {
            println!("DEBUG: 2y is zero, returning infinity");
            return Point::infinity();
        }
        
        let two_y_inv = match two_y.invert() {
            Ok(inv) => {
                println!("DEBUG: Successfully inverted 2y");
                inv
            },
            Err(e) => {
                println!("DEBUG: Failed to invert 2y: {:?}", e);
                println!("DEBUG: 2y = {:02x?}", two_y.data);
                return Point::infinity();
            }
        };
        
        let lambda = three_x_squared.mul(&two_y_inv);
        
        // x₃ = λ² - 2x₁
        let x3 = lambda.square().sub(&self.x.add(&self.x));
        
        // y₃ = λ(x₁ - x₃) - y₁
        let y3 = lambda.mul(&self.x.sub(&x3)).sub(&self.y);
        
        println!("DEBUG: Point::double completed successfully");
        Point { x: x3, y: y3, infinity: false }
    }
    
    // Scalar multiplication
    pub fn multiply(&self, scalar: &[u8; 32]) -> Self {
        println!("DEBUG: Point::multiply called with scalar {:02x?}", scalar);
        
        let mut result = Point::infinity();
        let mut addend = *self;
        
        // Process scalar bit by bit (little-endian)
        for (byte_idx, &byte) in scalar.iter().enumerate() {
            for bit in 0..8 {
                if (byte >> bit) & 1 == 1 {
                    println!("DEBUG: Adding at byte {} bit {}", byte_idx, bit);
                    result = result.add(&addend);
                }
                if byte_idx < 31 || bit < 7 { // Don't double on the last iteration
                    addend = addend.double();
                }
            }
        }
        
        println!("DEBUG: Point::multiply completed");
        result
    }
    
    // Compress point to 33 bytes
    pub fn compress(&self) -> [u8; 33] {
        if self.infinity { return [0u8; 33]; }
        
        let mut result = [0u8; 33];
        result[0] = if self.y.data[0] & 1 == 0 { 0x02 } else { 0x03 }; // Fixed: check LSB for little-endian
        result[1..33].copy_from_slice(&self.x.data);
        result
    }
    
    // Serialize uncompressed point
    pub fn serialize_uncompressed(&self) -> [u8; 65] {
        if self.infinity { return [0u8; 65]; }
        
        let mut result = [0u8; 65];
        result[0] = 0x04;
        result[1..33].copy_from_slice(&self.x.data);
        result[33..65].copy_from_slice(&self.y.data);
        result
    }
    
    // Decompress point from 33 bytes
    pub fn from_compressed(data: &[u8; 33]) -> Result<Self, Error> {
        if data.iter().all(|&b| b == 0) { return Ok(Point::infinity()); }
        
        let y_is_odd = match data[0] {
            0x02 => false,
            0x03 => true,
            _ => return Err(Error::InvalidPublicKey),
        };
        
        let mut x_bytes = [0u8; 32];
        x_bytes.copy_from_slice(&data[1..33]);
        let x = FieldElement::new(x_bytes)?;
        
        // y² = x³ + 7
        let y_squared = x.square().mul(&x).add(&{
            let mut seven = [0u8; 32];
            seven[0] = 7; // Fixed: little-endian
            FieldElement::new(seven).unwrap()
        });
        
        let y_candidate = y_squared.sqrt()?;
        let y = if (y_candidate.data[0] & 1 == 1) == y_is_odd { // Fixed: check LSB for little-endian
            y_candidate
        } else {
            y_candidate.negate()
        };
        
        Point::new(x, y)
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        if self.infinity && other.infinity { return true; }
        if self.infinity || other.infinity { return false; }
        self.x == other.x && self.y == other.y
    }
}

impl Eq for Point {}