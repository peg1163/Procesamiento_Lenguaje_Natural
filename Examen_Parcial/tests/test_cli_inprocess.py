import unittest, runpy, sys, json, tempfile
from pathlib import Path
from unittest.mock import patch

def run_mod(mod, argv):
    with patch.object(sys, "argv", [mod] + argv):
        runpy.run_module(mod, run_name="__main__", alter_sys=True)

class TestCLIInProcess(unittest.TestCase):
    def test_train_eval_bench_inprocess(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)

            # Corpus con suficientes líneas para que el split no quede vacío
            lines = ["abcabc(())+==", "a+a=", "b+b=", "c+c=", "()()", "abc()", "a(b)c="]
            (out / "corpus_raw.txt").write_text("\n".join(lines * 20) + "\n", encoding="utf-8")

            # train mini
            run_mod("src.train", ["--out", str(out), "--epochs", "1",
                                  "--lr", "0.0003", "--batch", "2", "--ctx", "32",
                                  "--pos", "rope", "--device", "cpu"])
            m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("train", m)
            self.assertGreater(len(m["train"]["loss"]), 0)

            # eval
            run_mod("src.eval", ["--out", str(out)])
            self.assertTrue((out / "eval.txt").exists())

            # bench mini
            run_mod("src.bench", ["--out", str(out), "--T", "8"])
            self.assertTrue((out / "bench.txt").exists())
