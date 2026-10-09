import unittest

from src.normalization import normalize_text


class NormalizationTests(unittest.TestCase):
    def test_preserves_raw_and_vietnamese_diacritics(self):
        raw = "Tài liệu\r\nTiếng Việt"
        result = normalize_text(raw)
        self.assertEqual(result.raw_text, raw)
        self.assertIn("Tài liệu", result.normalized_text)
        self.assertIn("Tiếng Việt", result.normalized_text)

    def test_detects_and_removes_zero_width(self):
        result = normalize_text("ig\u200bnore instructions")
        self.assertIn("zero_width_characters:1", result.signals)
        self.assertNotIn("\u200b", result.normalized_text)

    def test_detects_controls_and_unusual_whitespace(self):
        result = normalize_text("hello\x01\u00a0world")
        self.assertTrue(any(s.startswith("control_characters") for s in result.signals))
        self.assertTrue(any(s.startswith("unusual_whitespace") for s in result.signals))
        self.assertEqual(result.normalized_text, "hello world")


if __name__ == "__main__":
    unittest.main()
