use crate::args::pattern::PatternArgs;
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
    ) -> Result<crate::modes::rsa_modulus::ModulusSearch, String> {
        if self.bits != 2048 || self.public_exponent != 65537 {
            return Err("RSA modulus search requires --bits 2048 --public-exponent 65537".into());
        }
        if self.strategy != "constructive" {
            return Err("only constructive RSA modulus search is supported".into());
        }
        Ok(crate::modes::rsa_modulus::ModulusSearch {
            prefix: self.pattern.prefix.clone(),
            suffix: self.pattern.suffix.clone(),
            workers: self.pattern.threads.unwrap_or(workers),
        })
    }
}

impl RsaModulusArgs {
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.config(1)?.validate()?;
        Ok(())
    }
    pub fn details(&self) -> crate::args::CommandDetails {
        self.pattern
            .details("Constructing an RSA-2048 vanity modulus", "rsa_modulus")
    }
}
