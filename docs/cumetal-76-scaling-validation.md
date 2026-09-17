# CuMetal #76 proof-scaling consumer validation

## Result at `fb3644a`

[Draft PR #142](https://github.com/Lulzx/cuda-metal/pull/142) implements the later
[#76](https://github.com/Lulzx/cuda-metal/issues/76) address-demand/range-proof
corrections at `fb3644a251befdc4c1a9d82a8742ae04ae2498c4`, on #139/#138/#135.
The commit is pushed and GitGuardian passes at that exact head. The normal
consumer was rebuilt against its matched immutable Nix compiler/runtime.

| Check | Outcome |
| --- | --- |
| Matching CPU baseline | **53 pass:** Bitcoin 39, RSA-PSS 14 |
| LLVM7 Bitcoin self-tests | Translation rejects after **94.244 s**, next owner **#136** |
| LLVM7 RSA-PSS self-tests | Translation rejects after **80.859 s**, next owner **#141** |
| Full-module GPU launches | **0**; all 53 selected checks blocked before execution |

Both attempts finish within 600-second deadlines, report every selected check
once, and verify the expected source/compiler/runtime/PTX identities. The 39/14
CLI failures report their module's shared compilation failure; they are not
53 numerical mismatches or independent compiler invocations. Downstream
[#30](https://github.com/brandonros/vanity-miner-rs/issues/30) and
[#35](https://github.com/brandonros/vanity-miner-rs/issues/35) remain open.
Five workload trackers remain closed; 11 remain open. No full 32-combination
current-pin sweep or additional workload acceptance is claimed.

## Completed #76 stages and next owners

- **Bitcoin:** the previous retained-prefix-fact exhaustion is gone. Endpoint,
  affine-sibling, necessary Boolean and literal/sentinel-iterator proofs clear
  the preceding stores. The saved pointer cell is bytes `[256,264)` at PTX 34534.
  The next rejection is the field-inversion helper call 24791. Its complete
  transitive writes are `[496,537)`, disjoint from that cell; instantiating
  parameter-relative helper write footprints is existing
  [#136](https://github.com/Lulzx/cuda-metal/issues/136#issuecomment-5710663950).
  Other earlier calls still need proof after this one clears.
- **RSA-PSS:** lazy demanded-join expansion clears address-demand construction
  under the unchanged 1,000,000 work limit. Translation reaches direct `clz.b64`
  at PTX 55734, owned by [#141](https://github.com/Lulzx/cuda-metal/issues/141).

No proof budget was raised and no unknown value was assigned pointer provenance.
The [committed upstream report](https://github.com/brandonros/cuda-metal/blob/fb3644a251befdc4c1a9d82a8742ae04ae2498c4/docs/ptx-memory-proof-scaling-validation.md)
records bounds, tests, development compiler identities and full-input replays.
The development-compiler times 97.974 s / 80.260 s are separate from the normal
consumer times above. Neither pair measures isolated performance.

## Scoped regression checks

| Configuration | Units | PTX functional |
| --- | --- | --- |
| Release, shim OFF |57 pass/1 benchmark-precondition skip|57 pass/1 offline-tool skip|
| Debug, shim ON |60 pass / same skip|57 pass / same skip|

Apple M5 / macOS 26.6.2, LLVM/Clang 21.1.8. Two unit checks requiring unavailable
Apple reference/CLI prerequisites are excluded. Release's only initial failure
was a negative fixture that accidentally made its checked load unreachable;
the corrected fixture's focused rerun passes. Debug's complete selected suite
passes. These are scoped suites, not all-project acceptance.

Both configurations pass 16 numerical memory fixtures  × 65 lane positions and 29
compile-only refusal controls. CPU expectations check written bytes and saved
pointer payloads, with input/output guards and exact generic Apple GPU
provenance. Five of six added positive fixtures reject on the frozen parent;
the small literal64 iterator is existing-behavior regression coverage. These
small GPU checks do not establish the blocked full modules run.

#76's native legacy numerical gate still needs the unavailable offline Apple
Metal toolchain. Upstream PR integration remains outstanding. Those separate
gates are not unimplemented original conversion or proof-scaling fixes.

## Artifact identities

Consumer code base: `8d59d1f90deda1e16d3029952177d528cbe46e4b`; publication branch
HEAD before this update: `c7ad78efbba3761b45406519e1dd4ced51b23a2e`. Comparing all `logic/`,
`cli/src/` and these device entry sources against the PTX producer gives an
empty diff. Dependency manifests changed separately; original PTX is retained.
The normal all-mode CPU and CuMetal CLIs build under `nix develop .#cumetal` and
`cargo build --locked --release`; build results and commands are retained.

| Artifact | Identity |
| --- | --- |
| CuMetal commit |`fb3644a251befdc4c1a9d82a8742ae04ae2498c4`|
| Nix package |`/nix/store/dyggxmi8hrmyq44avgw1j8kpajc704h7-vanity-cumetal-fb3644a251be`|
| Nix source |`/nix/store/hgx0i22228hp0zimxgxwlxni9r72855l-source`|
| Source NAR |`sha256-jKNRcd++p88j2JDoNoYP7piycii5B3ETOQw7PY41LbY=`|
| Compiler SHA-256 |`698950d37c1f1c223579f5d40032c2609ba9a01b104530ae33e5dece3bd3183e`|
| Runtime SHA-256 |`ccc56ad0154edeb9a017ac6458ec7a1f1d88770347db1b67bab581fd69a92dfd`|
| CuMetal CLI SHA-256 |`c844ec47e08e4e72aa37c634220b608778168ec8bdd12584a7c128111a69193d`|
| CPU CLI SHA-256 |`aba99c302d9d7cde66d90373e94e08c46150607777cc6f77cafc3303aea7bb23`|
| PTX producer |`afe80210ea28748cc58c3ba75f877bfa4a6b1ecd`|
| LLVM7 Bitcoin PTX |`0bbd7000a8949ff1ca3776c938c9f72f14da477f388a13983bb8e5cce4bb25bc`|
| LLVM7 RSA-PSS PTX |`af511dc9a746971a79e062e0165c82f78fe7e9b2d20f83067fd387b64aed4407`|

PTX came from [Actions35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652).
The CLI receives original PTX via `--ptx` and the matched package via
`--cumetal-root`. `CUMETAL_TRACE_GPU=1` and
`CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0` are set. Inputs are unchanged
before/after; package/binary hashes match. Local evidence is retained in
`.cumetal-artifacts/issue-76-scaling-fb3644a/`: `metadata.json`, `results.json`,
`build-results.json`, exact input snapshots, all 203 inventory names, selected
names, CLI commands and logs. These local files are not public attachments.
