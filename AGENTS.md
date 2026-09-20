# Metal validation

Use the default `nix develop` shell. `rust-toolchain.toml` selects the stock Rust
producer, whose LLVM must be the major llvm-metal links and no newer. The flake
supplies that toolchain and `llvm-metalc`, which brings its own LLVM; nothing
else of llvm-metal's build belongs here. `flake.nix` pins the compiler and the
Cargo manifests pin its runtime crates: keep the revisions aligned. The NVPTX
target emits LLVM bitcode only.

- Build bundles with `scripts/build-metal.sh --mode <mode>`, a wrapper around
  `llvm-metalc build`. Each promoted kernel
  crate lives directly in `crates/kernels/<mode>` and has its own Cargo lockfile.
- Run searches through `scripts/run-metal.sh`. Default Cargo features use Metal;
  `--no-default-features` supports CPU references and non-Apple test hosts.
- Keep validation scoped to the actual source and artifacts. Record build manifests
  and logs under ignored `artifacts/`. Do not substitute old bundles for fresh
  source and attribute their results to the new revision.
- Test affected kernel contracts, host validation, and CPU references first, then
  compile the affected full kernels and run their ignored integration tests on an
  Apple GPU. Compilation alone is not a workload pass.
- Measure Rust/LLVM compilation, AIR lowering, Apple pipeline compilation, and GPU
  execution separately. Record cache use; reuse incremental builds deliberately.
- Keep one integrator in control of a checkout, pins, build directories, and GPU
  execution. Use separate worktrees for concurrent changes.

## Documentation

- Do not create per-task, per-artifact, planning, validation, or handoff Markdown
  files unless the user explicitly requests one. These reports are disposable
  and can be regenerated when needed.
- Keep raw logs, commands, hashes, and structured results in ignored artifact
  directories. Put durable technical context on existing issues/PRs when the
  user has authorized GitHub updates, rather than creating another Markdown file.
- After work is published, remove completed worktrees that have no unique work
  or active user. Preserve unpublished changes, branches, and commits.

## README maintenance

- Keep `README.md` a short entry point: purpose, supported modes, and essential
  setup, build, run, and self-test commands. Keep it under 100 lines unless the
  user explicitly requests a longer guide; do not cram paragraphs onto long lines.
- Edit it only when essential user-facing instructions change or become wrong.
  Completing an internal change or investigation is not a reason to add a section.
- Correct or replace existing instructions instead of appending updates. Use
  plain, direct wording; omit promotional language and repeated explanations.
- Do not add implementation narratives, debugging history, validation reports,
  timings, hashes, artifact inventories, agent handoffs, or migration chronicles.
- Use `--help` for exhaustive CLI options.
  Do not duplicate those details in the README or create extra Markdown files to
  hold material removed from it. Follow the documentation policy above.
- Before finishing a README edit, review the whole file for stale instructions,
  duplication, and unnecessary detail. Preserve the commands needed to get started.
