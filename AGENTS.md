# Metal validation

Use the default `nix develop` shell. `rust-toolchain.toml` selects the stock Rust
producer, whose LLVM must be the major llvm-metal links and no newer. The flake
supplies that toolchain and `llvm-metalc`, which brings its own LLVM; nothing
else of llvm-metal's build belongs here. `flake.nix` pins the compiler and the
Cargo manifests pin its runtime crates: keep the revisions aligned. The NVPTX
target emits LLVM bitcode only.

- Build bundles with `just build <mode>`, a wrapper around
  `llvm-metalc build`. Each promoted kernel
  crate lives directly in `crates/kernels/<mode>` and has its own Cargo lockfile.
- Run searches through `just run <mode>`. Default Cargo features use Metal;
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

## Tooling rules

Each of these removes something that was built here and had to be deleted.

- **No Python, and no second scripting language.** Logic is Rust: a `#[test]`, a
  CLI subcommand, or llvm-metal's builder. The `justfile` is the only task
  runner, and a recipe only sequences `cargo` and `llvm-metalc`. If a recipe
  needs parsing, conditions beyond a line, or state, the logic belongs in Rust.
  Do not add scripts.
- **Tests are `cargo test`.** No runner around the runner: no test inventories,
  coverage or ownership JSON, discovery cross-checks, per-test timeouts, or
  wrappers that select tests by name. A test that must not run by default is
  `#[ignore]` with its requirement in the reason. CI's job timeout is the timeout.
- **No metadata about the code that the code does not read.** No catalogs,
  manifests or registries kept in step by hand, and no checker whose job is to
  compare two hand-written lists. Derive the list (from a directory, a Cargo
  feature, a Rust const) or delete it.
- **llvm-metal's build is not this repository's concern.** `flake.nix` is the Rust
  toolchain plus the `llvm-metalc` package. Never name LLVM, `LLVM_SYS_*`,
  llvm-downgrade, `opt` pipelines or compiler source paths here. If building a
  kernel needs something new, it goes into `llvm-metalc build`.
- **Provenance is the builder's manifest.** Do not re-hash sources, re-verify
  bundles or record commands in scripts; `kernel.build.json` and the loader do it.
- **Fix forward.** When a toolchain bump breaks something, find the cause and fix
  it on the new toolchain; do not retreat to the old one to keep tests green.
  Delete a check that blocks work and guards nothing anyone would act on.

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
