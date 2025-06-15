// Error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    InvalidSecretKey,
    InvalidPublicKey,
    InvalidSignature,
    ArithmeticError,
}