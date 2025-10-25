import argparse, math, random
from pathlib import Path
from typing import Optional
import torch
from .mini_transformer import MiniTransformer
from .corpus import CharTokenizer, load_corpus, make_splits

p = argparse.ArgumentParser()
p.add_argument("--out", default="out")
p.add_argument("--ctx", type=int, default=None, help="longitud de contexto T para evaluar (opcional)")
args = p.parse_args()


lines = load_corpus(args.out)
_, va, _ = make_splits(lines)


tok = CharTokenizer()
model = MiniTransformer(len(tok.stoi))
model_path = Path(args.out) / "model.pt"
if not model_path.exists():
    raise FileNotFoundError("No existe out/model.pt. Ejecuta primero: python run.py train")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

def sample_seq(model, start: str = "1+2=", steps: int = 16, ctx: Optional[int] = None) -> str:
    idx = torch.tensor([tok.encode(start)], dtype=torch.long)
    for _ in range(steps):
        x = idx
        if ctx is not None and x.size(1) > ctx:
            x = x[:, -ctx:] 
        with torch.no_grad():
            logits = model(x)
        nxt = torch.distributions.Categorical(logits=logits[:, -1, :]).sample().unsqueeze(0)
        idx = torch.cat([idx, nxt], dim=1)
    return tok.decode(idx[0].tolist())


N = 32
loss_sum = 0
tokens = 0
T = args.ctx

if T is None:
    for _ in range(N):
        s = random.choice(va) + "\n"
        ids = tok.encode(s)
        if len(ids) < 2:
            continue
        x = torch.tensor(ids[:-1]).unsqueeze(0)
        y = torch.tensor(ids[1:]).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        loss_sum += loss
        tokens += len(ids) - 1
else:
    for _ in range(N):
        s = random.choice(va) + "\n"
        ids = tok.encode(s)
        for i in range(0, len(ids) - 1, T):
            chunk = ids[i:i+T]
            if len(chunk) < 2:
                continue
            x = torch.tensor(chunk[:-1]).unsqueeze(0)
            y = torch.tensor(chunk[1:]).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1)).item()
            loss_sum += loss
            tokens += len(chunk) - 1

ppl = math.exp(loss_sum / max(1, tokens))
Path(args.out).mkdir(exist_ok=True)
(Path(args.out) / "eval.txt").write_text(
    f"perplexity: {ppl:.3f}\nexample: {sample_seq(model, ctx=T)}\n",
    encoding="utf-8"
)
print("perplexity", ppl)
