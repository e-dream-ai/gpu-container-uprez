import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.input_validator import InputValidator


class ValidateUpscaleFactorTest(unittest.TestCase):
    def _assert_accepts(self, value, expected):
        result = InputValidator.validate_upscale_factor(value)
        self.assertTrue(result["valid"], msg=result.get("errors"))
        self.assertEqual(result["value"], expected)

    def _assert_rejects(self, value):
        result = InputValidator.validate_upscale_factor(value)
        self.assertFalse(result["valid"])
        self.assertTrue(result["errors"])

    def test_accepts_supported_factors(self):
        self._assert_accepts(1, 1)
        self._assert_accepts(2, 2)
        self._assert_accepts(4, 4)

    def test_coerces_numeric_strings(self):
        self._assert_accepts("4", 4)

    def test_rejects_unsupported_factors(self):
        self._assert_rejects(3)
        self._assert_rejects(0)
        self._assert_rejects(8)

    def test_rejects_non_integer(self):
        self._assert_rejects("abc")
        self._assert_rejects(None)


if __name__ == "__main__":
    unittest.main()
