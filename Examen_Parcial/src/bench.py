import argparse, os, time
from pathlib import Path
import torch
from .attn import CausalSelfAttention

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

p = argparse.ArgumentParser()
p.add_argument("--out", default="out")
p.add_argument("--T", type=int, default=256)          # <= era 512
p.add_argument("--d_model", type=int, default=128)
p.add_argument("--n_heads", type=int, default=4)
args = p.parse_args()

B, T, D, H = 1, args.T, args.d_model, args.n_heads
x = torch.randn(B, T, D)


att_full = CausalSelfAttention(D, H, kv_cache=False)
t0 = time.time()
with torch.no_grad():
    _ = att_full(x)
t_full = time.time() - t0


att_step = CausalSelfAttention(D, H, kv_cache=True)
ys = []
t0 = time.time()
with torch.no_grad():
    for t in range(T):
        y_last = att_step(x[:, t:t+1, :], use_cache=True)
        ys.append(y_last)              # (B,1,D)
y = torch.cat(ys, dim=1)               # (B,T,D)
t_cache = time.time() - t0

Path(args.out).mkdir(exist_ok=True)
(Path(args.out) / "bench.txt").write_text(f"full={t_full:.6f}s\ncache={t_cache:.6f}s\n")
print("full", t_full, "cache", t_cache)
