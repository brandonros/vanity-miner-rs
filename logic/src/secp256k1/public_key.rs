use super::point::Point;
use super::error::Error;
use super::secret_key::SecretKey;
use super::field_element::FieldElement;

// Public key wrapper
#[derive(Debug, Clone, Copy)]
pub struct PublicKey {
    point: Point,
}

impl PublicKey {
    pub fn from_secret_key(secret_key: &SecretKey) -> Self {
        let generator = Point::generator();
        let point = generator.multiply(&secret_key.data);
        Self { point }
    }
    
    pub fn serialize(&self) -> [u8; 33] {
        self.point.compress()
    }
    
    pub fn serialize_uncompressed(&self) -> [u8; 65] {
        self.point.serialize_uncompressed()
    }
    
    pub fn from_slice(data: &[u8]) -> Result<Self, Error> {
        match data.len() {
            33 => {
                let mut compressed = [0u8; 33];
                compressed.copy_from_slice(data);
                let point = Point::from_compressed(&compressed)?;
                Ok(Self { point })
            }
            65 => {
                if data[0] != 0x04 {
                    return Err(Error::InvalidPublicKey);
                }
                
                let mut x_bytes = [0u8; 32];
                let mut y_bytes = [0u8; 32];
                x_bytes.copy_from_slice(&data[1..33]);
                y_bytes.copy_from_slice(&data[33..65]);
                
                let x = FieldElement::new(x_bytes)?;
                let y = FieldElement::new(y_bytes)?;
                let point = Point::new(x, y)?;
                
                Ok(Self { point })
            }
            _ => Err(Error::InvalidPublicKey),
        }
    }
}