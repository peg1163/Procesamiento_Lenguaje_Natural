import unittest
import json
from pathlib import Path

class TestIdempotenciaArtefactos(unittest.TestCase):
    def test_metrics_json_exists_and_has_train(self):
        p = Path("out/metrics.json")
        if not p.exists():
            self.skipTest("Ejecuta primero: make train")
        m = json.loads(p.read_text())
        self.assertIn("train", m)
        self.assertIn("step", m["train"])
        self.assertIn("loss", m["train"])

    def test_model_file_exists(self):
        p = Path("out/model.pt")
        if not p.exists():
            self.skipTest("Ejecuta primero: make train")
        self.assertTrue(p.stat().st_size > 0)

if __name__ == "__main__":
    unittest.main()
