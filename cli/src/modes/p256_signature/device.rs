use super::*;

impl Prepared<'_> {
    pub(super) fn search_device(
        &self,
        control: &SearchControl,
        device: &mut EvaluateBatch<'_>,
    ) -> Result<Option<Winner>, String> {
        let config = self.config;
        let pattern = &self.pattern;
        let private = &self.private;
        let public = self.public;
        let original = &self.original;
        let digest = self.digest;
        let seed = &self.seed;
        let candidate_limit = self.candidate_limit;
        use logic::modes::p256_signature::P256SignatureRequest;
        let (source, offset, length) = match config.source {
            SearchSource::Message { offset, length } => (0, offset as u64, length as u64),
            SearchSource::Ephemeral => (1, 0, 0),
        };
        let request = Zeroizing::new(P256SignatureRequest {
            private: **private,
            seed: **seed,
            fingerprint: Sha256::digest(public),
            digest,
            worker: 0,
            offset,
            length,
            source,
            target: match config.target {
                SignatureTarget::Raw => 0,
                SignatureTarget::R => 1,
                SignatureTarget::S => 2,
            },
            s_form: match config.s_form {
                SForm::Low => 0,
                SForm::High => 1,
                SForm::Either => 2,
            },
            reserved: 0,
        });
        if control.continuous_device() {
            crate::runner::batches::stream(
                |start, count| device(&request, pattern, original, start, count),
                candidate_limit,
                control,
                |counter, bytes| {
                    self.format_winner(Winner {
                        counter,
                        worker: 0,
                        signature: bytes[..64].try_into().expect("fixed signature width"),
                    })
                },
            )?;
            return Ok(None);
        }
        let found = crate::runner::batches::find(
            |start, count| device(&request, pattern, original, start, count),
            candidate_limit,
            control,
            |counter, bytes| {
                let winner = Winner {
                    counter,
                    worker: 0,
                    signature: bytes[..64].try_into().expect("fixed signature width"),
                };
                self.verify_winner(&winner).map(|_| true)
            },
        )?;
        Ok(found.map(|(counter, result)| Winner {
            counter,
            worker: 0,
            signature: result.bytes[..64]
                .try_into()
                .expect("fixed signature width"),
        }))
    }
}
