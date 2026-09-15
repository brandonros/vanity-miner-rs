use super::*;

impl Prepared<'_> {
    pub(super) fn search_device(
        &self,
        control: &SearchControl,
        device: &mut EvaluateBatch<'_>,
    ) -> Result<Option<Winner>, String> {
        let config = self.config;
        let pattern = &self.pattern;
        let seed = &self.seed;
        let deriver = &self.deriver;
        use logic::modes::p256_public_key::P256PublicRequest;
        let request = Zeroizing::new(P256PublicRequest {
            seed: **seed,
            worker: 0,
            target: match config.target {
                PublicTarget::X => 0,
                PublicTarget::Y => 1,
                PublicTarget::Xy => 2,
                PublicTarget::Uncompressed => 3,
            },
            reserved: 0,
        });
        let mut winner = None;
        if control.continuous_device() {
            crate::runner::batches::stream(
                |start, count| device(&request, pattern, &[], start, count),
                u64::MAX,
                control,
                |counter, bytes| {
                    let private = candidate_scalar(deriver, 0, counter as u128)
                        .ok_or("device scalar reconstruction failed")?;
                    self.format_winner(Winner {
                        private,
                        public: bytes[..65].try_into().expect("fixed point width"),
                    })
                },
            )?;
            return Ok(None);
        }
        let found = crate::runner::batches::find(
            |start, count| device(&request, pattern, &[], start, count),
            u64::MAX,
            control,
            |counter, bytes| {
                let private = candidate_scalar(deriver, 0, counter as u128)
                    .ok_or("device scalar reconstruction failed")?;
                let public = bytes[..65]
                    .try_into()
                    .map_err(|_| "invalid device point width")?;
                let candidate = Winner { private, public };
                if !verify_winner(&candidate, config.target, pattern) {
                    return Err("device P-256 public key failed verification".into());
                }
                winner = Some(candidate);
                Ok(true)
            },
        )?;
        Ok(if found.is_some() { winner } else { None })
    }
}
