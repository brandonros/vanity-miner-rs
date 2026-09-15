# CuMetal status — 2026-09-15

Retested **2026-09-15, 18:36–18:59 UTC**, using CuMetal **`e5acf8cc0c65`**,
matching `flake.lock` at validation time. **True = passed GPU validation.**
**False = failed translation or did not finish within the 300-second limit.**

This is the last complete run, not a result for the later `92a9b8f4de23` or
`7d12f120a6b8` locks.
See the [issue ownership matrix](cumetal-issue-matrix.md) for upstream fix status
and subsequent targeted research.

**3 true, 13 false: 11 translation failures and 2 timed-out production modes.**

| Module | Works in CuMetal | Issue |
|---|---|---|
| Solana — production | **False** | [#23](https://github.com/brandonros/vanity-miner-rs/issues/23) |
| Bitcoin — production | **False** | [#24](https://github.com/brandonros/vanity-miner-rs/issues/24) |
| Ethereum — production | **False** | [#25](https://github.com/brandonros/vanity-miner-rs/issues/25) |
| Shallenge — production | **True** | — |
| P-256 public key — production | **True** | — |
| P-256 signature — production | **False** | [#26](https://github.com/brandonros/vanity-miner-rs/issues/26) |
| RSA modulus — production | **False** | [#27](https://github.com/brandonros/vanity-miner-rs/issues/27) |
| RSA-PSS — production | **False** | [#28](https://github.com/brandonros/vanity-miner-rs/issues/28) |
| Solana — self-test | **False** | [#29](https://github.com/brandonros/vanity-miner-rs/issues/29) |
| Bitcoin — self-test | **False** | [#30](https://github.com/brandonros/vanity-miner-rs/issues/30) |
| Ethereum — self-test | **False** | [#31](https://github.com/brandonros/vanity-miner-rs/issues/31) |
| Shallenge — self-test | **True** | — |
| P-256 public key — self-test | **False** | [#32](https://github.com/brandonros/vanity-miner-rs/issues/32) |
| P-256 signature — self-test | **False** | [#33](https://github.com/brandonros/vanity-miner-rs/issues/33) |
| RSA modulus — self-test | **False** | [#34](https://github.com/brandonros/vanity-miner-rs/issues/34) |
| RSA-PSS — self-test | **False** | [#35](https://github.com/brandonros/vanity-miner-rs/issues/35) |

RSA modulus and RSA-PSS production timed out. Both RSA-PSS search variants now
translate to Metal successfully; neither completed GPU validation. RSA modulus
was sampled waiting for a GPU command to complete. All seven failing self-test
groups were blocked by translation; together the self-tests reported **8 passed,
152 failed, 0 skipped**.

The previous `98cf505` run also had 3 true / 13 false, but its RSA-PSS production
failed translation. The unchanged boolean totals hide that progress.

See the [full validation report](cumetal-validation.md) for diagnostics, coverage,
source revisions, and artifact hashes.
