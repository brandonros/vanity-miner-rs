use super::*;

impl Prepared<'_> {
    /// Reconstruct and independently verify before claiming or printing a winner.
    pub(super) fn verify_winner(&self, winner: &Winner) -> Result<(Vec<u8>, Vec<u8>), String> {
        let config = self.config;
        let pattern = &self.pattern;
        let key = &self.key;
        let public = &self.public;
        let original = &self.original;
        let base_salt = &self.base_salt;
        let mut message = original.clone();
        let mut salt = base_salt.clone();
        apply_candidate(
            &config.source,
            base_salt,
            winner.counter,
            &mut message,
            &mut salt,
        )?;
        let digest: [u8; 32] = Sha256::digest(&message);
        let reproduced = sign_explicit_salt(key, &digest, &salt)?;
        if reproduced != winner.signature || !pattern.matches(&reproduced) {
            return Err("RSA-PSS winner failed reconstruction".into());
        }
        public
            .verify(
                Pss::new_with_salt::<Sha256>(salt.len()),
                &digest,
                &reproduced,
            )
            .map_err(|_| "RSA-PSS final verification failed")?;
        Ok((message, salt))
    }

    pub(super) fn format_winner(&self, winner: Winner) -> Result<String, String> {
        let (message, salt) = self.verify_winner(&winner)?;
        let public = &self.public;
        Ok(format!(
            "[rsa-pss] public_key={}\n[rsa-pss] signature={}\n[rsa-pss] salt={}\n[rsa-pss] message={}",
            hex::encode(public.n().to_bytes_be()),
            hex::encode(winner.signature),
            hex::encode(&salt),
            hex::encode(&message),
        ))
    }
}
