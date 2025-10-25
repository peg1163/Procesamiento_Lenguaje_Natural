import sys, subprocess, json
from pathlib import Path

OUT = Path("out")

def run(*args):
    return subprocess.run([sys.executable, *args], check=True, capture_output=True, text=True)

def test_tiny_train_and_eval():
    OUT.mkdir(exist_ok=True)
    # entrenamiento mini
    run("-m", "src.train", "--out", str(OUT),
        "--epochs", "1", "--lr", "0.0003", "--batch", "2", "--ctx", "32",
        "--pos", "rope", "--device", "cpu")
    # métricas
    m = json.loads((OUT/"metrics.json").read_text(encoding="utf-8"))
    assert "train" in m and len(m["train"]["loss"]) > 0
    # eval
    run("-m", "src.eval", "--out", str(OUT))
    txt = (OUT/"eval.txt").read_text(encoding="utf-8").lower()
    assert "perplexity" in txt
