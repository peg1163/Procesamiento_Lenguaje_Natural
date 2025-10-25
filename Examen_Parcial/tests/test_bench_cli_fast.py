import sys, subprocess
from pathlib import Path

def test_bench_tiny():
    out = Path("out"); out.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "src.bench", "--out", str(out), "--T", "8"],
                   check=True, capture_output=True, text=True)
    assert (out/"bench.txt").exists()
