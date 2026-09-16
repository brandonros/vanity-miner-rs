"""Entry coverage must not silently accept an old multi-entry RSA artifact."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "compile_ptx", Path(__file__).resolve().parents[1] / "compile-ptx-cumetal.py"
)
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


class EntryCoverage(unittest.TestCase):
    def test_only_the_expected_entry_is_accepted(self):
        expected = compiler.PRODUCTION_ENTRIES["rsa_modulus"]
        with tempfile.TemporaryDirectory() as directory:
            ptx = Path(directory) / "rsa_modulus.ptx"
            ptx.write_text("// .entry ignored()\n.visible .entry " + expected + "() {}")
            compiler.require_single_entry(ptx, expected)
            for source in (
                ".entry kernel_rsa_advance() {}",
                ".entry " + expected + "() {}\n.entry extra() {}",
                "// no entries",
            ):
                with self.subTest(source=source):
                    ptx.write_text(source)
                    with self.assertRaisesRegex(SystemExit, "rebuild the PTX bundle"):
                        compiler.require_single_entry(ptx, expected)

    def test_report_names_the_selected_entry_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "failure.log").write_text("fixture failure")
            metadata = dict(scope="fixture", revision="fixture", entries=compiler.PRODUCTION_ENTRIES)
            rows = [(21, "rsa_modulus", 1.25, "PASS", "ok.log"),
                    (7, "rsa_modulus", 2.5, "FAIL", "failure.log")]
            compiler.write_report(run, metadata, rows)
            report = (run / "report.md").read_text()
            self.assertEqual(report.count("`kernel_rsa_modulus_vanity`"), 2)
            self.assertIn("fixture failure", report)


if __name__ == "__main__":
    unittest.main()
