import unittest, tempfile
from pathlib import Path
from src.utils import set_seed, device_select, save_json
from src.corpus import CharTokenizer, load_corpus, make_splits

class TestUtilsCorpus(unittest.TestCase):
    def test_utils_and_corpus_basic(self):
        set_seed(123)
        dev = device_select("auto")
        dtype = getattr(dev, "type", str(dev))
        self.assertIn(dtype, ("cpu", "cuda", "mps"))

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)

            # --- corpus con suficientes líneas para que val/test no queden vacíos ---
            base = ["abcabc(())+==", "abc", "a+a=", "b+b=", "c+c=", "()()", "abc()", "a(b)c="]
            text = "\n".join(base * 50) + "\n"   # 400 líneas aprox.
            (out / "corpus_raw.txt").write_text(text, encoding="utf-8")

            lines = load_corpus(str(out))     # lista de líneas
            raw_text = "\n".join(lines)       # string para el tokenizer

            tok = CharTokenizer(raw_text)
            ids = tok.encode("abc")
            self.assertEqual(tok.decode(ids), "abc")

            tr, va, te = make_splits(lines)   # usa los defaults del repo
            self.assertGreater(len(tr), 0)
            self.assertGreater(len(va), 0)    # ahora sí: no debería ser 0
            self.assertGreater(len(te), 0)

            j = out / "z.json"
            save_json({"ok": True}, j)
            self.assertTrue(j.exists())

