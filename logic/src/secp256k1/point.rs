use super::field_element::FieldElement;
use super::error::Error;
use super::constants;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    x: FieldElement,
    y: FieldElement,
    infinity: bool,
}

impl Point {
    pub fn new(x: FieldElement, y: FieldElement) -> Result<Self, Error> {
        let point = Self { x, y, infinity: false };
        if point.is_on_curve() {
            Ok(point)
        } else {
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
            data[31] = 7;
            FieldElement::new(data).unwrap()
        };
        let right_side = x_cubed.add(&seven);
        
        y_squared == right_side
    }
    
    // Point addition using the group law
    pub fn add(&self, other: &Self) -> Self {
        // Handle point at infinity cases
        if self.infinity {
            return *other;
        }
        if other.infinity {
            return *self;
        }
        
        // Handle point doubling
        if self.x == other.x {
            if self.y == other.y {
                return self.double();
            } else {
                // Points are inverses, result is infinity
                return Point::infinity();
            }
        }
        
        // General case: P + Q where P != Q
        // slope = (y2 - y1) / (x2 - x1)
        // x3 = slope^2 - x1 - x2
        // y3 = slope * (x1 - x3) - y1
        
        todo!("Implement point addition")
    }
    
    pub fn double(&self) -> Self {
        if self.infinity {
            return *self;
        }
        
        // Point doubling: 2P
        // slope = (3 * x1^2) / (2 * y1)  [derivative of curve equation]
        // x3 = slope^2 - 2 * x1
        // y3 = slope * (x1 - x3) - y1
        
        todo!("Implement point doubling")
    }
    
    // Scalar multiplication: k * P
    pub fn multiply(&self, scalar: &[u8; 32]) -> Self {
        // Use double-and-add algorithm
        let mut result = Point::infinity();
        let mut addend = *self;
        
        for byte in scalar.iter().rev() {
            for bit in 0..8 {
                if (byte >> bit) & 1 == 1 {
                    result = result.add(&addend);
                }
                addend = addend.double();
            }
        }
        
        result
    }
    
    // Compress point to 33 bytes (0x02/0x03 + x coordinate)
    pub fn compress(&self) -> [u8; 33] {
        if self.infinity {
            return [0u8; 33];
        }
        
        let mut result = [0u8; 33];
        
        // First byte indicates y-coordinate parity
        result[0] = if self.y.data[31] & 1 == 0 { 0x02 } else { 0x03 };
        
        // Copy x-coordinate
        result[1..33].copy_from_slice(&self.x.data);
        
        result
    }
    
    // Serialize uncompressed point to 65 bytes (0x04 + x + y)
    pub fn serialize_uncompressed(&self) -> [u8; 65] {
        if self.infinity {
            return [0u8; 65];
        }
        
        let mut result = [0u8; 65];
        result[0] = 0x04;
        result[1..33].copy_from_slice(&self.x.data);
        result[33..65].copy_from_slice(&self.y.data);
        
        result
    }
    
    // Decompress point from 33 bytes
    pub fn from_compressed(data: &[u8; 33]) -> Result<Self, Error> {
        if data[0] == 0x00 {
            return Ok(Point::infinity());
        }
        
        if data[0] != 0x02 && data[0] != 0x03 {
            return Err(Error::InvalidPublicKey);
        }
        
        let mut x_bytes = [0u8; 32];
        x_bytes.copy_from_slice(&data[1..33]);
        let x = FieldElement::new(x_bytes)?;
        
        // Calculate y^2 = x^3 + 7
        let y_squared = x.square().mul(&x).add(&{
            let mut seven = [0u8; 32];
            seven[31] = 7;
            FieldElement::new(seven).unwrap()
        });
        
        // Find square root
        let y_candidate = y_squared.sqrt()?;
        
        // Choose correct y based on parity
        let y = if (y_candidate.data[31] & 1) == (data[0] & 1) {
            y_candidate
        } else {
            // Negate y_candidate
            todo!("Implement field negation")
        };
        
        Point::new(x, y)
    }
}