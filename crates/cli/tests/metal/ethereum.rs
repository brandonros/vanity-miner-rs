#![cfg(all(feature = "metal", feature = "ethereum", target_os = "macos"))]
use logic::search::vanity::BytePattern;
use std::path::PathBuf;
use vanity_miner::modes::ethereum::metal::EthereumTransport;

fn artifacts() -> PathBuf {
    super::support::artifacts("VANITY_METAL_ETHEREUM_ARTIFACTS", "ethereum")
}

#[test]
#[ignore = "build just build ethereum first; requires Apple GPU"]
fn candidates_patterns_winners_and_errors_match_cpu() {
    super::support::address::check::<EthereumTransport>(
        &artifacts(),
        super::support::address::Cases {
            seed: 10088153575472065218,
            patterns: [
                BytePattern::new(&[], &[]).unwrap(),
                BytePattern::new(&[0; 21], &[]).unwrap(),
                BytePattern::new(&[0x55], &[0x02]).unwrap(),
                BytePattern::new(&[], &[0xff]).unwrap(),
            ],
            address: &hex::decode("55e56b7b70dc37a7a1419e1e84ea4e6e237ef602").unwrap(),
            winner_lane: 0,
            check_next_batch: false,
            extra_groups: &[],
        },
    );
}

#[test]
#[ignore = "build just build ethereum first; requires Apple GPU"]
fn bounded_cli_prints_verified_winners_and_completes_misses() {
    for (prefix, suffix, matches) in [
        ("", "", true),
        ("55e56b7b70dc37a7a1419e1e84ea4e6e237ef602", "", true),
        ("00000000000000000000000000000000000000000000", "", false),
    ] {
        let stdout = super::support::address::cli(
            &artifacts(),
            "ethereum-vanity",
            10088153575472065218,
            prefix,
            suffix,
        );
        assert_eq!(
            stdout.contains("[ethereum] address=0x"),
            matches,
            "{stdout}"
        );
        if prefix.starts_with("55") {
            assert!(
                stdout.contains("[ethereum] address=0x55e56b7b70dc37a7a1419e1e84ea4e6e237ef602")
            );
        }
    }
}
