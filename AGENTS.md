# CuMetal validation

`flake.lock` selects the CuMetal fork revision and source hash used for validation.
The `cumetal` Nix package builds its compiler and runtime together, including the
pinned VF64 submodule. Local contribution worktrees are not build inputs.

- Use `nix develop .#cumetal --command cargo ...` with the `cumetal` Cargo feature.
  This builds the dependency and embeds its absolute package path in the CLI.
  `--cumetal-root` can select another output of `nix build .#cumetal`; the CLI
  verifies its manifest against the consumer's locked revision and checks both
  artifact hashes before loading the runtime.
- When additional contribution changes are requested, inspect the branch history
  and update the CuMetal flake input/lock deliberately. Directory names such as
  `cuda-metal-upstream` do not identify which patches a checkout contains. Verify
  the locked commit actually contains the requested changes before reporting results.
- Use PTX inputs. The CLI translates a snapshot through the pinned compiler;
  independently supplied compiler/runtime binaries and precompiled Metal modules
  cannot bypass the dependency selection.
- Record the source revision, compiler/runtime paths and hashes, and PTX path/hash
  printed by the CLI with validation results. Also identify the vanity-miner
  revision that produced the PTX.
- If the selected build cannot compile, load, or execute, report that exact
  failure. Do not substitute an older build and attribute its results to the
  selected source.

Keep validation claims scoped to the recorded source and artifacts. Historical
results must be labeled historical when the tested dependency revision changes.

## Development loop

- Iterate on small reproducers, refusal cases, and focused CPU/GPU tests. Replay
  an affected full module after its small tests pass; batch shared regression
  checks and the matched Nix integration build after a coherent group of fixes.
- Give one integrator control of the active checkout, build directory, pin, and
  GPU runs. Parallelize independent research or edits to separately owned files.
  Reuse incremental builds; relink affected test executables before running them.
- Measure PTX translation, Metal compilation, pipeline creation, and GPU execution
  separately. A faster rejection or smaller source is not a workload pass.
- Keep acceptance runs fresh and record any cache use explicitly. Never attribute
  a cached result to a different compiler, input, entry, or set of options.

## Documentation

- Keep `docs/cumetal-issue-matrix.md` as the single concise CuMetal status summary.
  Update current rows; do not append historical reports or repeated evidence.
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
- Use `--help` for exhaustive CLI options and the issue matrix for CuMetal status.
  Do not duplicate those details in the README or create extra Markdown files to
  hold material removed from it. Follow the documentation policy above.
- Before finishing a README edit, review the whole file for stale instructions,
  duplication, and unnecessary detail. Preserve the commands needed to get started.
