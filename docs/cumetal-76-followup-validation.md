# #76 follow-up: normal-miner validation at `13efc29`

Measured 2026-09-16 using the rebuilt normal miner and immutable matched Nix
compiler/runtime. The original #76 failures clear, but **none of these three
self-test combinations reaches GPU execution**. Downstream #29 and #37 stay open.
All four unchanged CPU groups were rerun: **147/147 checks pass**.

| Self-test module | Version | Seconds | Next translation blocker | Upstream owner |
| --- | --- | ---: | --- | --- |
| shallenge | llvm7 | 25.994 | Mixed-vector pointer reload before address use9917 | [#118](https://github.com/Lulzx/cuda-metal/issues/118) |
| solana | llvm7 | 88.168 | Helper call at46229 may write pointer cell loaded at48114 | [#136](https://github.com/Lulzx/cuda-metal/issues/136) |
| shallenge | llvm21 | 13.549 | Undefined `%rd16` at `$L__BB18_1`, zero-marker guard | [#130](https://github.com/Lulzx/cuda-metal/issues/130) |

All three processes exit 1 before their 600-second deadline, with 121 checks blocked
by translation and zero GPU launches. The CLI reports each blocked assertion as
`FAIL`; these are not 121 observed numerical mismatches. Input, compiler/runtime
and CLI hashes were checked. PTX inputs are unchanged from producer `afe80210`
and [Actions 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652).

## Exact identities

- CuMetal commit: `13efc293c6cd06da5616a87b2dd8db81b1b0e0df`.
- Miner host: `80699f24bd477b5551bed1d8eff3bc910a833210`.
- Compiler SHA-256: `52f642906af96dbe3617fba51afae43cd381c4dc684dcf14bba4e1f606fa91e5`.
- Runtime SHA-256: `2827bb1939ddfc5668f49e9dbc5d653c549f9a4291f8f2049b794d92c505a87d`.
- CuMetal CLI SHA-256: `99306009052b865c7a266437f80d6daa3d391b595b818b7e9bce887f18673e6f`.
- CPU CLI SHA-256: `cbeb9593afaf1c257915c6ca7f83b4326b045b366f20269b46031aed9dd528f7`.

- llvm7 / shallenge PTX: `a2e120bb765bfc5b90bea18ae2cf964843ea3a25e39d171316cae545f71c5f3b`.
- llvm7 / solana PTX: `d33cba71e185acd397a245f9b95908abc1c2f5d694d6ebb551ecbf4d4aef9344`.
- llvm21 / shallenge PTX: `de774d6d87edec30418f1ec76eb0499a635e159e73686e00b97d6df21ecf098c`.

The CPU executable is the retained unchanged-source CPU build; it was rerun with
its recorded hash. The GPU executable was rebuilt against the new pinned pair.
[Commands, phase outcomes and logs](../.cumetal-artifacts/issue-76-13efc29/results.json),
[identities](../.cumetal-artifacts/issue-76-13efc29/metadata.json) and
[runner](../.cumetal-artifacts/issue-76-13efc29/run.py) are retained locally.

## Compiler acceptance is separate

[Draft PR #135](https://github.com/Lulzx/cuda-metal/pull/135) implements the remaining
supported #76 proofs and conversion contracts. Both development configurations
pass 55 runnable PTX functional tests; native legacy GPU acceptance still requires
the offline Apple toolchain. GitGuardian passed on the exact signed head.
[The compiler acceptance document](https://github.com/brandonros/cuda-metal/blob/13efc293c6cd06da5616a87b2dd8db81b1b0e0df/docs/issue-76-acceptance-tests.md)
records baseline controls, numerical coverage, budgets and exclusions.

[#137](https://github.com/Lulzx/cuda-metal/issues/137) separately tracks the
preexisting escaped-cell validation omission found by negative compiler controls.
That omitted validation has no demonstrated GPU numerical outcome. The new helper
write-footprint owner #136 has a public small reproducer, but the full Solana
callee's transitive effects still need proof before selecting a correction.
