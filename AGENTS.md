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
