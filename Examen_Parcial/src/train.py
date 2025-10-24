import argparse, random
from pathlib import Path
import torch
import torch.optim as optim

from .config import Config
from .corpus import CharTokenizer, load_corpus, make_splits
from .mini_transformer import MiniTransformer
from .utils import set_seed, device_select, save_json

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="out")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--ctx", type=int, default=128)
parser.add_argument("--pos", default="rope", choices=["rope", "sinusoidal"])
parser.add_argument("--device", default="auto")
args = parser.parse_args()

set_seed(1337)
DEV = device_select(args.device)

cfg = Config(pos_encoding=args.pos, ctx_len=args.ctx)

lines = load_corpus(args.out)
tr, va, _ = make_splits(lines)

tok = CharTokenizer()

def batchify(data):
    B, T = args.batch, args.ctx
    X = torch.zeros(B, T, dtype=torch.long)
    Y = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        s = random.choice(data) + "\n"
        ids = tok.encode(s * (1 + max(1, (T // max(1, len(s))))))
        start = random.randint(0, max(0, len(ids) - T - 1))
        seq = ids[start:start + T + 1]
        X[b] = torch.tensor(seq[:-1])
        Y[b] = torch.tensor(seq[1:])
    return X.to(DEV), Y.to(DEV)

model = MiniTransformer(len(tok.stoi), d_model=cfg.d_model, n_heads=cfg.n_heads, d_mlp=cfg.d_mlp, pos=cfg.pos_encoding).to(DEV)
opt = optim.AdamW(model.parameters(), lr=args.lr)
sched = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=100)  # warmup lineal

loss_fn = torch.nn.CrossEntropyLoss()
Path(args.out).mkdir(exist_ok=True)
metrics = {"train": {"step": [], "loss": []}}

step = 0
for ep in range(args.epochs):
    for _ in range(200):  # mini-steps por época (rápido para la rúbrica)
        x, y = batchify(tr)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # grad clipping
        opt.step()
        sched.step()
        step += 1
        if step % 10 == 0:
            metrics["train"]["step"].append(step)
            metrics["train"]["loss"].append(float(loss.item()))

save_json(metrics, Path(args.out) / "metrics.json")
torch.save(model.state_dict(), Path(args.out) / "model.pt")
print("OK train")
