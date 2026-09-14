# New-mode integration architecture review

Scope: the four RSA/P-256 modes, their CLI runners, CUDA transport, shared logic,
and self-tests. This is a source review, not a claim that every repository design
issue has been found. The baseline is the direct per-mode design documented in
`cli/src/modes/mod.rs`.

## Main finding: extra mode-dispatch framework — fixed locally

`cli/src/device_search.rs` combined a cross-mode request enum, a device-search
trait, batching/winner control, and a CPU test adapter. Every mode was enumerated
again in both the host adapter and CUDA transport.

The enum and trait are removed. Each host mode declares its own typed batch
callback. `search_batches.rs` handles only bounded batches, cancellation, and
verified winner selection. `test_support.rs` contains host evaluation adapters
and compiles only for tests/self-test. CUDA transport exposes concrete entry
methods without a request enum. The common logic result record is now named
`candidate_result.rs`.

## Five additional structural inconsistencies

| Finding | Evidence and consequence | Status |
| --- | --- | --- |
| Grouped kernel and candidate modules | `crypto_vanity.rs` hid four entry points behind a macro; the old logic `device_search.rs` owned all four request/evaluator pairs. This broke the established per-mode source layout. | Fixed locally: four matching kernel/logic files, explicit kernel entry points. |
| New CLI modes bypassed the mode directory | CPU command wrappers lived in `crypto_runner.rs`; GPU command setup lived inside the runner's extra classification/match block. | Fixed locally: CPU/GPU entry points in `cli/src/modes/{p256_public,p256_signature,rsa_modulus,rsa_pss_search}.rs`. Shared session reporting is in `common/search_session.rs`; CUDA batches are under `runner/`. |
| Two CUDA resource/scheduling paths | `common/gpu_context.rs` owns legacy launch configuration; `runner/cuda_batches.rs` separately owns streams, stack settings, and sequential rotation across devices. Legacy dispatch uses per-device threads. | Fixed: one runner-owned context/module/stream per device, one worker scheduler for all modes, shared cancellation and joined workers; transport only borrows resources. Batch sizes remain workload-specific. |
| Two statistics/session models | `main.rs` creates `GlobalStats` for every command, while the new modes count candidates through `SearchControl` and report through `common/search_session.rs`. The new paths do not consume the supplied legacy stats. | Fixed: one library `GlobalStats` shared by runners and `SearchControl`; counters, timing, and reporting use that object, with explicit candidate units. |
| Separate self-test inventories | Legacy tests use the numbered `SELF_TEST_*` inventory; crypto fixtures use separate per-mode differential functions and group messages. CuMetal's generated test inventory covers only the legacy entries. | Fixed: one `self_test_suite` inventory and PASS/FAIL/SKIP report for CPU, CUDA, and CuMetal, including existing crypto differential fixtures. CuMetal explicitly skips unsupported crypto transport; dedicated additional primitive kernels are a separate coverage improvement. |

The separate RSA/P-256 arithmetic helpers, protected output publication, host
winner reconstruction, and secret-buffer erasure are justified responsibilities,
not mismatches merely because they are shared or differ from blockchain output.

## Validation boundary

CPU tests and native feature checks exercise the refactor locally. They do not
establish CUDA/PTX compilation or NVIDIA execution for the edited source. The
previously merged revision's green CUDA jobs do not validate these local edits.
