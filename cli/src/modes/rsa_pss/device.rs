use super::*;

impl Prepared<'_> {
    pub(super) fn search_device(
        &self,
        control: &SearchControl,
        device: &mut EvaluateBatch<'_>,
    ) -> Result<Option<Winner>, String> {
        let config = self.config;
        let pattern = &self.pattern;
        let key = &self.key;
        let original = &self.original;
        let digest = self.digest;
        let base_salt = &self.base_salt;
        let limit = self.limit;
        use logic::modes::rsa_pss::RsaPssRequest;
        use rsa::traits::PrivateKeyParts;
        let (source, offset, length) = match config.source {
            PssSource::Salt { .. } => (0, 0, 0),
            PssSource::Message { offset, length, .. } => (1, offset as u64, length as u64),
        };
        let coefficient =
            Zeroizing::new(key.crt_coefficient().ok_or("missing RSA CRT coefficient")?);
        let mut request = Zeroizing::new(RsaPssRequest {
            p: *fixed_bytes(&key.primes()[0])?,
            q: *fixed_bytes(&key.primes()[1])?,
            dp: *fixed_bytes(key.dp().ok_or("missing RSA dp")?)?,
            dq: *fixed_bytes(key.dq().ok_or("missing RSA dq")?)?,
            q_inv: *fixed_bytes(&coefficient)?,
            digest,
            salt: [0; 222],
            reserved: [0; 2],
            offset,
            length,
            source,
            salt_length: base_salt.len() as u32,
        });
        request.salt[..base_salt.len()].copy_from_slice(base_salt);
        if control.continuous_device() {
            crate::runner::batches::stream(
                |start, count| device(&request, pattern, original, start, count),
                limit,
                control,
                |counter, bytes| {
                    self.format_winner(Winner {
                        counter,
                        signature: *bytes,
                    })
                },
            )?;
            return Ok(None);
        }
        let found = crate::runner::batches::find(
            |start, count| device(&request, pattern, original, start, count),
            limit,
            control,
            |counter, bytes| {
                let winner = Winner {
                    counter,
                    signature: *bytes,
                };
                self.verify_winner(&winner).map(|_| true)
            },
        )?;
        Ok(found.map(|(counter, result)| Winner {
            counter,
            signature: result.bytes,
        }))
    }
}
