# Preventing constant-result GPU self-tests

The full PTX artifact from run 34774932262 had 14 actual self-tests reduced to
pointer setup, a constant `1` store, and `ret`. The separate launch stub is
intentionally constant. Such checks do not exercise their intended operations
on the device.

The affected checks now pass their inputs through `core::hint::black_box`
**before** the tested operation. This follows the existing arithmetic probes
and preserves the kernel ABI and known-answer vectors. Hash equality uses two
independent opaque arrays, not a comparison of an array with itself. The
nonce-length slot directly tests `shallenge_nonce_len` with an opaque username
length; the separate hash and nonce checks cover the other work.

This is a best-effort optimization barrier, not a language-level guarantee.
`inline(never)` alone already failed to prevent folding of the slice-last test.
Putting a barrier around a result boolean would not preserve the computation.
Host-supplied input buffers and host-checked output buffers remain the stronger
future design for a variable-input test suite.

The all-features workflow runs `scripts/verify-ptx-self-tests.py` on its actual
release PTX and includes `self-test-audit.json` in the artifact. The gate rejects
the observed pointer-setup/constant-store pattern, including constant failure,
for every self-test except `kernel_self_test_stub`. It does not analyze helper
calls, arbitrary PTX control flow, or partial folding; a passing gate is not a
proof of full algorithm coverage or successful device execution. The existing
entry inventory gate separately checks that all expected kernels exist.

Local checks:

```sh
cargo test -p logic --all-features --release --locked
python3 scripts/test-verify-ptx-entries.py
python3 scripts/test-verify-ptx-self-tests.py
python3 scripts/verify-ptx-self-tests.py path/to/output.ptx
```

The original artifact must fail the last command with the 14 known folded
checks. A new artifact must pass, and the affected kernel bodies must be
inspected for input-dependent computation and result predicates.
