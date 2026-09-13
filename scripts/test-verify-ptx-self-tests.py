#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('audit', Path(__file__).with_name('verify-ptx-self-tests.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def entry(name='sha256', value='1', extra=''):
    return f'''.visible .entry kernel_self_test_{name}(.param .u64 output) {{
    .reg .b32 %r<3>; .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [output]; cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, {value}; {extra}
    st.global.u32 [%rd2+32], %r1; ret;
    }}'''


class AuditTests(unittest.TestCase):
    def test_success_and_failure_constants(self):
        for value in ('0', '1'):
            self.assertEqual(module.audit(entry(value=value))['constant_result_tests'], ['kernel_self_test_sha256'])

    def test_stub_only_exemption(self):
        self.assertEqual(module.audit(entry('stub'))['constant_result_tests'], [])

    def test_computation_is_not_classified_as_constant(self):
        self.assertEqual(module.audit(entry(extra='xor.b32 %r1, %r1, %r2;'))['constant_result_tests'], [])

    def test_comments_and_register_copy(self):
        self.assertEqual(module.audit(entry(value='2', extra='/* { } */ // }\n mov.u32 %r2, 1; mov.u32 %r1, %r2;'))['constant_result_tests'], ['kernel_self_test_sha256'])

    def test_missing_or_malformed_entries(self):
        for text in ('', entry()[:-1], entry()+entry()):
            with self.assertRaises(ValueError):
                module.audit(text)


if __name__ == '__main__':
    unittest.main()
