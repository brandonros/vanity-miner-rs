use super::*;

impl Prepared<'_> {
    pub(super) fn format_winner(&self, winner: Winner) -> Result<String, String> {
        let config = self.config;
        let pattern = &self.pattern;
        // Recheck immediately before printing the matched key.
        if !verify_winner(&winner, config.target, pattern) {
            return Err("P-256 winner failed final verification".into());
        }
        Ok(format!(
            "[p256-public-key] public_key={}\n[p256-public-key] sec1_public_key={}\n[p256-public-key] private_key={}",
            hex::encode(config.target.bytes(&winner.public)),
            hex::encode(winner.public),
            hex::encode(*winner.private),
        ))
    }
}

/// Host reconstruction: validate SEC1/on-curve encoding, derive the public point
/// again from the scalar, and check both complete byte equality and the pattern.
pub(super) fn verify_winner(winner: &Winner, target: PublicTarget, pattern: &HexPattern) -> bool {
    use p256::elliptic_curve::sec1::ToEncodedPoint;
    let Ok(private) = SecretKey::from_slice(winner.private.as_ref()) else {
        return false;
    };
    let Ok(public) = p256::PublicKey::from_sec1_bytes(&winner.public) else {
        return false;
    };
    // P-256 has cofactor one; a valid nonidentity public point has prime order.
    let derived = private.public_key();
    derived == public
        && derived.to_encoded_point(false).as_bytes() == winner.public
        && pattern.matches(target.bytes(&winner.public))
}
