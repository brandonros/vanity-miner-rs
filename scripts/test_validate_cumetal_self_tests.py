"""The report gate must reject partial or duplicated evidence, even with exit 0."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'validator', Path(__file__).with_name('validate-cumetal-self-tests.py'))
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'kernels/src/self_test.rs'
        self.source = source.read_text()
        self.entries = validator.inventory(self.source)
        self.lines = []
        for name, slot in self.entries:
            self.lines += [f'CUMETAL_PROVENANCE event=kernel_launch kernel="{name}" '
                           'source=generic_ptx launch_success=true duration_ns=1',
                           f'NUMERICAL_PASS kernel={name} slot={slot}; other slots and guards intact']

    def test_complete_evidence(self):
        self.assertEqual(len(self.entries), 119)
        observed, errors = validator.validate_output('\n'.join(self.lines), self.entries)
        self.assertEqual(observed, self.entries)
        self.assertEqual(errors, [])

    def test_missing_or_duplicate_or_wrong_slot_fails(self):
        variants = [self.lines[:-1], self.lines + self.lines[-1:],
                    [line.replace('slot=117;', 'slot=116;') for line in self.lines]]
        for lines in variants:
            with self.subTest(lines=len(lines)):
                self.assertTrue(validator.validate_output('\n'.join(lines), self.entries)[1])

    def test_success_flags_without_gpu_launches_fail(self):
        lines = [line for line in self.lines if line.startswith('NUMERICAL_PASS')]
        self.assertTrue(validator.validate_output('\n'.join(lines), self.entries)[1])

    def test_missing_or_duplicate_source_slot_fails(self):
        for source in [self.source.replace('results[117]', 'results[116]'), '']:
            with self.assertRaises(ValueError):
                validator.inventory(source)


if __name__ == '__main__':
    unittest.main()
