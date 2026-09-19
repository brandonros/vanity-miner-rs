use crate::args::pattern::PatternArgs;
use clap::Args;
use clap::ValueEnum;

#[derive(ValueEnum, Clone, Copy)]
pub enum PublicTarget {
    X,
    Y,
    Xy,
    Uncompressed,
}

#[derive(Args, Clone)]
pub struct P256PublicArgs {
    #[command(flatten)]
    pub pattern: PatternArgs,
    #[arg(long, value_enum, default_value = "xy")]
    pub target: PublicTarget,
}

impl P256PublicArgs {
    pub fn config(&self, workers: usize) -> crate::modes::p256_public_key::PublicKeySearch {
        use logic::crypto::p256::PublicTarget as Target;
        crate::modes::p256_public_key::PublicKeySearch {
            prefix: self.pattern.prefix.clone(),
            suffix: self.pattern.suffix.clone(),
            target: match self.target {
                PublicTarget::X => Target::X,
                PublicTarget::Y => Target::Y,
                PublicTarget::Xy => Target::Xy,
                PublicTarget::Uncompressed => Target::Uncompressed,
            },
            workers: workers,
        }
    }
}

impl P256PublicArgs {
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.config(1).validate()?;
        Ok(())
    }
    pub fn details(&self) -> crate::args::CommandDetails {
        self.pattern.details("Searching NIST P-256 public points")
    }
}
