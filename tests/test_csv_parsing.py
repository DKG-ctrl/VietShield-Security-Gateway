import csv
from pathlib import Path
import tempfile
import unittest

from src.gateway import SecurityGateway


class StubDetector:
    model_name = "test_stub"

    def score_text(self, text: str) -> float:
        return 0.1


class CSVParsingTests(unittest.TestCase):
    def test_quoted_comma_multiline_quotes_and_unicode(self):
        content = 'Id,text\n1,"Xin chào, Việt Nam"\n2,"Dòng một\nDòng hai, có dấu phẩy và ""quote"""\n'
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source = Path(directory) / "input.csv"
            output = Path(directory) / "output.csv"
            source.write_text(content, encoding="utf-8")
            with source.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["text"], "Xin chào, Việt Nam")
            self.assertIn("\n", rows[1]["text"])
            self.assertIn('"quote"', rows[1]["text"])
            SecurityGateway(StubDetector()).scan_batch(source, output)
            with output.open("r", encoding="utf-8", newline="") as handle:
                produced = list(csv.DictReader(handle))
            self.assertEqual([row["Id"] for row in produced], ["1", "2"])


if __name__ == "__main__":
    unittest.main()
