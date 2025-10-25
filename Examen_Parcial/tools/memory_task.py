# tools/memory_task.py
import argparse, random, math, csv
from pathlib import Path
import torch, torch.nn as nn
from src.attn import CausalSelfAttention

def gen_parens(n, max_T):
    def balanced():
        s, bal = [], 0
        T = random.randint(2, max_T // 2) * 2
        for _ in range(T):
            if bal == 0:
                s.append(0); bal += 1
            else:
                if random.random() < 0.5 and len(s) < T - bal:
                    s.append(0); bal += 1
                else:
                    s.append(1); bal -= 1
        while bal > 0:
            s.append(1); bal -= 1
        return s[:max_T], 1
    def unbalanced():
        s, _ = balanced()
        if s:
            i = random.randrange(len(s)); s[i] ^= 1
        return s[:max_T], 0
    data = []
    for _ in range(n):
        if random.random() < 0.5: data.append(balanced())
        else: data.append(unbalanced())
    return data

class GRUClassifier(nn.Module):
    def __init__(self, d_model=64, vocab=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.fc  = nn.Linear(d_model, 2)
    def forward(self, x):
        e = self.emb(x)
        _, h = self.gru(e)       # h: (1,B,D)
        h = h[-1]                # (B,D)
        return self.fc(h)        # (B,2)

class AttnClassifier(nn.Module):
    def __init__(self, d_model=64, n_heads=4, vocab=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, kv_cache=False)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 2)
    def forward(self, x):
        e = self.emb(x)              # (B,T,D)
        h = self.attn(e)             # (B,T,D)
        h = self.ln(h)
        h_last = h[:, -1, :]         # usar último token
        return self.fc(h_last)       # (B,2)

def to_batches(data, batch, max_T):
    
    X, y = [], []
    for s, lab in data:
        z = s[:max_T] + [0]*(max_T - len(s))
        X.append(z); y.append(lab)
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    for i in range(0, len(X), batch):
        yield X[i:i+batch], y[i:i+batch]

def train_eval(model, train, val, max_T, epochs=3, lr=3e-3, batch=64, device="cpu"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in to_batches(train, batch, max_T):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in to_batches(val, batch, max_T):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmin(dim=1) if False else model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct/total if total else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--ntrain", type=int, default=2000)
    ap.add_argument("--nval", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(123); torch.manual_seed(123)

    train = gen_parens(args.ntrain, args.T)
    val   = gen_parens(args.nval,   args.T)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    
    rnn = GRUClassifier(d_model=64, vocab=2)
    acc_rnn = train_eval(rnn, train, val, args.T, epochs=args.epochs, lr=3e-3, batch=64, device=dev)

    
    att = AttnClassifier(d_model=64, n_heads=4, vocab=2)
    acc_att = train_eval(att, train, val, args.T, epochs=args.epochs, lr=3e-3, batch=64, device=dev)

    csv_path = out / "memory_accuracy.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["T","model","accuracy"])
        w.writerow([args.T, "RNN(GRU)", f"{acc_rnn:.4f}"])
        w.writerow([args.T, "Attention(1L)", f"{acc_att:.4f}"])
    print(f"[memory] guardado {csv_path}")

if __name__ == "__main__":
    main()
