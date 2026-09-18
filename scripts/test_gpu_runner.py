"""Host-only regressions for fail-closed GPU discovery (no Metal or Nix required)."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('gpu_runner', Path(__file__).with_name('test-gpu.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class DiscoveryTests(unittest.TestCase):
    def test_private_gpu_test_and_explicit_cpu_exclusion(self):
        runner.check_discovery('lib', 'session: test\nmanual_search: test\n',
                               [{'target': 'lib', 'name': 'session'}],
                               [{'target': 'lib', 'name': 'manual_search'}])

    def test_new_target_or_missing_test_cannot_silently_pass(self):
        for target, output, cases in [
            ('new_gpu_target', 'unlisted: test\n', []),
            ('lib', '', [{'target': 'lib', 'name': 'session'}]),
            ('lib', 'session: test\nnew_session: test\n', [{'target': 'lib', 'name': 'session'}]),
        ]:
            with self.subTest(target=target, output=output), self.assertRaisesRegex(RuntimeError, 'discovery differs'):
                runner.check_discovery(target, output, cases, [])

    def test_duplicates_and_exclusions_cannot_hide_gpu_tests(self):
        case = {'target': 'lib', 'name': 'session'}
        for cases, exclusions in [([case, case], []), ([case], [case])]:
            with self.assertRaisesRegex(RuntimeError, 'duplicate or excluded'):
                runner.check_discovery('lib', 'session: test\n', cases, exclusions)

if __name__ == '__main__':
    unittest.main()
