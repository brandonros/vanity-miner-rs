use crate::common::pattern_args::PatternArgs;
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
    pub fn config(&self, workers: usize) -> vanity_miner::search::p256_public_key::PublicKeySearch {
        use logic::crypto::p256::PublicTarget as Target;
        vanity_miner::search::p256_public_key::PublicKeySearch {
            prefix: self.pattern.prefix.clone(),
            suffix: self.pattern.suffix.clone(),
            target: match self.target {
                PublicTarget::X => Target::X,
                PublicTarget::Y => Target::Y,
                PublicTarget::Xy => Target::Xy,
                PublicTarget::Uncompressed => Target::Uncompressed,
            },
            workers: self.pattern.threads.unwrap_or(workers),
        }
    }
}
