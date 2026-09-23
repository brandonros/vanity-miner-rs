use super::{Case, inventory};
use clap::Args;

#[derive(Args, Clone, Default)]
pub struct SelfTestArgs {
    /// Run named checks (repeat this option to select more than one).
    /// Each selected mode runs in full; only the selected checks are reported.
    #[arg(long = "check", value_name = "MODE.CHECK")]
    pub checks: Vec<String>,
    /// List check names and descriptions without initializing a device.
    #[arg(long)]
    pub list: bool,
}

impl SelfTestArgs {
    pub fn selected(&self) -> Result<Vec<Case>, String> {
        let inventory = inventory();
        for name in &self.checks {
            if !inventory.iter().any(|case| case.name == *name) {
                return Err(format!("unknown self-test '{name}'; use self-test --list"));
            }
        }
        Ok(inventory
            .into_iter()
            .filter(|case| self.checks.is_empty() || self.checks.contains(&case.name))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{Cli, Command};
    use clap::Parser;

    fn self_test_args(command: Command) -> SelfTestArgs {
        match command {
            Command::SelfTest(args) => args,
            #[allow(unreachable_patterns)]
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn selection_uses_names_in_inventory_order_and_deduplicates() {
        let cases = inventory();
        let first = cases.first().unwrap().name.as_str();
        let last = cases.last().unwrap().name.as_str();
        let cli = Cli::try_parse_from([
            "vanity-miner",
            "self-test",
            "--check",
            last,
            "--check",
            first,
            "--check",
            first,
        ])
        .unwrap();
        let args = self_test_args(cli.command);
        let selected = args.selected().unwrap();
        assert_eq!(selected.len(), if first == last { 1 } else { 2 });
        assert_eq!(selected[0].name, first);
        assert_eq!(selected.last().unwrap().name, last);
    }

    #[test]
    fn unknown_and_numeric_selectors_are_rejected() {
        for name in ["185", "p256_public_key.typo"] {
            let error = SelfTestArgs {
                checks: vec![name.into()],
                list: false,
            }
            .selected()
            .unwrap_err();
            assert!(error.contains("unknown self-test"));
        }
        assert!(
            Cli::try_parse_from(["vanity-miner", "self-test", "--self-test-slot", "185"]).is_err()
        );
    }

    #[test]
    fn list_is_available_without_backend_options() {
        let cli = Cli::try_parse_from(["vanity-miner", "self-test", "--list"]).unwrap();
        let args = self_test_args(cli.command);
        assert!(args.list);
        assert_eq!(args.selected().unwrap().len(), inventory().len());
    }
}
