#!/usr/bin/env python3
"""Time all 32 PTX-to-MSL attempts with the CuMetal package selected by flake.lock.

First run: nix build .#cumetal --out-link .cumetal-artifacts/toolchain
Then run:  python3 scripts/compile-ptx-cumetal.py
"""
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import tarfile
import tempfile
import time


PRODUCTION_ENTRIES = {
    "solana": "kernel_solana_vanity",
    "bitcoin": "kernel_bitcoin_vanity",
    "ethereum": "kernel_ethereum_vanity",
    "shallenge": "kernel_shallenge",
    "rsa_modulus": "kernel_rsa_modulus_vanity",
    "rsa_pss": "kernel_rsa_pss_signature_vanity",
    "p256_public_key": "kernel_p256_public_key_vanity",
    "p256_signature": "kernel_p256_signature_vanity",
}


def require_single_entry(ptx, expected):
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx.read_text(), flags=re.S)
    entries = re.findall(r"\.entry\s+([A-Za-z_$][\w$]*)\s*\(", source)
    if entries != [expected]:
        raise SystemExit(f"{ptx}: expected only {expected}, found {entries}; rebuild the PTX bundle")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(run, metadata, rows):
    report = ["# CuMetal PTX translation timings", "", metadata["scope"], "",
              f"CuMetal revision: `{metadata['revision']}`. See [metadata](metadata.json) for provenance and hashes.", ""]
    for llvm in (7, 21):
        group = [row for row in rows if row[0] == llvm]
        passed = sum(row[3] == "PASS" for row in group)
        failed = sum(row[3] == "FAIL" for row in group)
        timed_out = sum(row[3] == "TIMEOUT" for row in group)
        elapsed = sum(row[2] for row in group)
        report.append(f"- LLVM {llvm}: {passed} passed, {failed} failed, {timed_out} timed out; {elapsed:.3f} seconds across attempts.")
    report += ["", "Times include failed attempts; totals are not a successful-compilation speed comparison.", "",
               "| LLVM | Module | Entry | Seconds | Result | Log |", "|---|---|---|---:|---|---|"]
    for llvm, module, seconds, status, log in rows:
        report.append(f"| {llvm} | {module} | `{metadata['entries'][module]}` | {seconds:.3f} | {status} | [log]({log}) |")
    report += ["", "## Failure diagnostics", ""]
    for llvm, module, seconds, status, log in rows:
        if status == "PASS":
            continue
        diagnostic = (run / log).read_text(errors="replace").strip()
        report += [f"### LLVM {llvm}: {module}", "", "```text",
                   diagnostic or status, "```", ""]
    (run / "report.md").write_text("\n".join(report) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=300,
                        help="seconds allowed per module (default: 300)")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    root = Path(__file__).resolve().parents[1]
    toolchain = (root / ".cumetal-artifacts/toolchain").resolve()
    manifest = json.loads((toolchain / "share/cumetal/build.json").read_text())
    locked = json.loads((root / "flake.lock").read_text())["nodes"]["cumetal"]["locked"]["rev"]
    if manifest["revision"] != locked:
        raise SystemExit("CuMetal revision mismatch; rebuild with nix build .#cumetal")
    compiler = toolchain / "bin/cumetalc"
    runtime = toolchain / "lib/libcumetal.dylib"
    for path, key in [(compiler, "compiler_sha256"), (runtime, "runtime_sha256")]:
        if sha256(path) != manifest[key]:
            raise SystemExit(f"CuMetal hash mismatch: {path}")

    run = Path(tempfile.mkdtemp(prefix="ptx-compile-", dir=root / ".cumetal-artifacts"))
    metadata = dict(manifest, compiler=str(compiler), runtime=str(runtime),
                    timeout_seconds=args.timeout, started_utc=time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    scope="serial PTX-to-MSL translation only; no Metal binary compilation or execution",
                    consumer_revision=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    provenance = root / "artifacts/ptx-timing-run.json"
    if provenance.exists():
        metadata["ptx_build_record"] = json.loads(provenance.read_text())
    metadata["bundles"] = {}
    entries = dict(PRODUCTION_ENTRIES)
    entries.update({"self_test_" + mode: "kernel_self_test_" + mode
                    for mode in PRODUCTION_ENTRIES})
    metadata["entries"] = entries
    modules = sorted(entries)
    # Snapshot only the expected files, avoiding stale PTX and unsafe archive paths.
    for llvm in (7, 21):
        bundle = root / f"artifacts/ptx-bundle-llvm{llvm}.tar.gz"
        metadata["bundles"][str(llvm)] = dict(path=str(bundle), sha256=sha256(bundle))
        dest = run / f"llvm{llvm}"
        dest.mkdir()
        with tarfile.open(bundle) as archive:
            for module in modules:
                member = archive.getmember(module + ".ptx")
                if not member.isfile() or not member.size:
                    raise SystemExit(f"Invalid PTX member: {member.name}")
                ptx = dest / member.name
                ptx.write_bytes(archive.extractfile(member).read())
                require_single_entry(ptx, entries[module])
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results: {run}", flush=True)
    rows = []
    with (run / "timings.tsv").open("w") as table:
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["llvm", "module", "entry", "seconds", "status", "exit_code", "ptx_sha256", "log"])
        for llvm in (7, 21):
            for module in modules:
                dest = run / f"llvm{llvm}"
                ptx, output, log = [dest / (module + ext) for ext in (".ptx", ".metal", ".log")]
                command = [str(compiler), str(ptx), "--entry", entries[module],
                           "--backend=cumetal-ir", "--ptx-strict",
                           "--overwrite", "--emit=msl", "-o", str(output)]
                print(f"Starting LLVM {llvm}: {module}", flush=True)
                started = time.monotonic()
                with log.open("w") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    try:
                        code = process.wait(timeout=args.timeout)
                        status = "PASS" if code == 0 and output.is_file() and output.stat().st_size else "FAIL"
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        code = process.wait()
                        status = "TIMEOUT"
                    except BaseException:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise
                seconds = time.monotonic() - started
                relative_log = str(log.relative_to(run))
                writer.writerow([llvm, module, entries[module], f"{seconds:.3f}", status, code, sha256(ptx), relative_log])
                table.flush()
                rows.append((llvm, module, seconds, status, relative_log))
                print(f"{status:7} LLVM {llvm:2} {module:30} {seconds:9.3f}s", flush=True)

    write_report(run, metadata, rows)
    print(f"Report: {run / 'report.md'}", flush=True)
    return int(any(row[3] != "PASS" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
