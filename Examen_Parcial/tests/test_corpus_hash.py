import unittest
from pathlib import Path

class TestCorpusHash(unittest.TestCase):
    def test_sha_present_and_length(self):
        sha = Path("out/CORPUS_SHA256.txt")
        if not sha.exists():
            self.skipTest("Ejecuta primero: make data")
        h = sha.read_text().strip()
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h.lower()))

    def test_corpus_raw_exists(self):
        raw = Path("out/corpus_raw.txt")
        if not raw.exists():
            self.skipTest("Ejecuta primero: make data")
        self.assertTrue(raw.stat().st_size > 0)

if __name__ == "__main__":
    unittest.main()
