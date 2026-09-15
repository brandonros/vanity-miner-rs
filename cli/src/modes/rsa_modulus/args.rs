use crate::common::pattern_args::PatternArgs;
use clap::Args;

#[derive(Args, Clone)]
pub struct RsaModulusArgs {
    #[command(flatten)]
    pub pattern: PatternArgs,
    #[arg(long, default_value_t = 2048)]
    pub bits: usize,
    #[arg(long, default_value_t = 65537)]
    pub public_exponent: u32,
    #[arg(long, default_value = "constructive", value_parser = ["constructive"])]
    pub strategy: String,
}

impl RsaModulusArgs {
    pub fn config(
        &self,
        workers: usize,
    ) -> Result<vanity_miner::search::rsa_modulus::ModulusSearch, String> {
        if self.bits != 2048 || self.public_exponent != 65537 {
            return Err("RSA modulus search requires --bits 2048 --public-exponent 65537".into());
        }
        if self.strategy != "constructive" {
            return Err("only constructive RSA modulus search is supported".into());
        }
        Ok(vanity_miner::search::rsa_modulus::ModulusSearch {
            prefix: self.pattern.prefix.clone(),
            suffix: self.pattern.suffix.clone(),
            workers: self.pattern.threads.unwrap_or(workers),
        })
    }
}
