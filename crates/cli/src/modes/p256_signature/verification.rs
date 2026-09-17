use super::*;

impl Prepared<'_> {
    /// Reconstruct and independently verify before claiming or printing a winner.
    pub(super) fn verify_winner(&self, winner: &Winner) -> Result<Vec<u8>, String> {
        let config = self.config;
        let pattern = &self.pattern;
        let private = &self.private;
        let public = self.public;
        let original = &self.original;
        let digest = self.digest;
        let deriver = &self.deriver;
        // Reconstruct from metadata instead of trusting worker message buffers.
        let mut message = original.clone();
        let reproduced = match config.source {
            SearchSource::Message { offset, length } => {
                write_message_counter(&mut message, offset, length, winner.counter as u128)
                    .map_err(|e| e.to_string())?;
                let raw = signatures::sign_message(private, &message)
                    .ok_or("winner signing reconstruction failed")?;
                signatures::matching_representation(&raw, config.target, config.s_form, pattern)
            }
            SearchSource::Ephemeral => {
                let nonce = candidate_scalar(deriver, winner.worker, winner.counter as u128)
                    .ok_or("winner nonce reconstruction failed")?;
                signatures::matching_ephemeral_signature(
                    private,
                    &digest,
                    &nonce,
                    config.target,
                    config.s_form,
                    pattern,
                )
            }
        };
        if reproduced != Some(winner.signature)
            || !signatures::verify(&public, &message, &winner.signature)
        {
            return Err("P-256 winner failed final reconstruction and verification".into());
        }
        Ok(message)
    }

    pub(super) fn format_winner(&self, winner: Winner) -> Result<String, String> {
        let message = self.verify_winner(&winner)?;
        let public = self.public;
        Ok(format!(
            "[p256-signature] public_key={}\n[p256-signature] signature={}\n[p256-signature] message={}",
            hex::encode(&public[1..]),
            hex::encode(winner.signature),
            hex::encode(&message),
        ))
    }
}
