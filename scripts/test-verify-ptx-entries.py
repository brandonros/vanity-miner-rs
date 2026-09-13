#!/usr/bin/env python3
import pathlib
import runpy
import tempfile
import unittest

manifest = runpy.run_path(str(pathlib.Path(__file__).with_name('verify-ptx-entries.py')))['manifest']


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.source = self.root / 'src'
        self.source.mkdir()
        (self.source / 'test.rs').write_text(
            '#[kernel]\n#[allow(dead_code)]\npub unsafe extern "C" fn kernel_a() {}\n'
            '#[kernel]\npub unsafe extern "C" fn kernel_self_test_stub() {}\n')
        self.ptx = self.root / 'test.ptx'

    def check_ptx(self, entries):
        self.ptx.write_text('\n'.join(f'.visible .entry {name}() {{ ret; }}' for name in entries))
        return manifest(self.ptx, self.source)

    def test_complete_inventory(self):
        result = self.check_ptx(['kernel_self_test_stub', 'kernel_a'])
        self.assertEqual(result['entry_count'], 2)
        self.assertEqual(result['self_test_count'], 1)
        self.assertEqual(result['entries'], ['kernel_a', 'kernel_self_test_stub'])

    def test_partial_build_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing='):
            self.check_ptx(['kernel_a'])

    def test_duplicate_entry_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicates=\\['kernel_a'\\]"):
            self.check_ptx(['kernel_a', 'kernel_a', 'kernel_self_test_stub'])

    def test_unexpected_entry_rejected(self):
        with self.assertRaisesRegex(ValueError, "extra=\\['unexpected'\\]"):
            self.check_ptx(['kernel_a', 'kernel_self_test_stub', 'unexpected'])


if __name__ == '__main__':
    unittest.main()
