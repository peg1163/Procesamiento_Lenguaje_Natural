# src/bench.py
import argparse, os, time, csv
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .attn import CausalSelfAttention
from .utils import device_select  # usa tu helper (auto/cpu/cuda/mps)

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

# ------------------------ Helpers comunes ------------------------

def maybe_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def timeit_ms(fn, reps=3, warmup=1, device=None):
    # warmup
    for _ in range(warmup):
        fn(); maybe_sync(device)
    # medir
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(); maybe_sync(device)
        ts.append((time.perf_counter() - t0) * 1000.0)
    mean = sum(ts)/len(ts)
    var  = sum((t-mean)**2 for t in ts)/len(ts)
    return mean, var**0.5  # mean, std

def time_and_peak_mem(fn, reps=3, warmup=1, device=None):
    # Devuelve latencia media/STD y pico de memoria (en bytes; None si no-GPU)
    # warmup
    for _ in range(warmup):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        fn(); maybe_sync(device)
    ts = []; peak = 0
    for _ in range(reps):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        fn(); maybe_sync(device)
        ts.append((time.perf_counter() - t0) * 1000.0)
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated(device))
    mean = sum(ts)/len(ts)
    var  = sum((t-mean)**2 for t in ts)/len(ts)
    std  = var**0.5
    return mean, std, (peak if device.type == "cuda" else None)

# ------------------------ Variantes (para comparación) ------------------------

class MLPBlock(nn.Module):
    """Bloque multi-linear simple por token (C -> 4C -> C)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model, bias=True),
            nn.GELU(),
            nn.Linear(4*d_model, d_model, bias=True),
        )
    def forward(self, x):
        return self.net(x)

# ------------------------ Bench principal ------------------------

def bench_default(args):
    """Mantiene compatibilidad con run.py: full + cache + step."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    device = device_select(args.device)
    B, T, D, H = 1, args.T, args.d_model, args.n_heads
    x = torch.randn(B, T, D, device=device)

    # Una sola instancia base y clonamos pesos
    base = CausalSelfAttention(D, H, kv_cache=True).to(device).eval()

    def clone(kv_cache: bool):
        m = CausalSelfAttention(D, H, kv_cache=kv_cache).to(device).eval()
        m.load_state_dict(base.state_dict())
        return m

    # FULL
    att_full = clone(kv_cache=False)
    def full_fn():
        with torch.no_grad():
            _ = att_full(x)
    full_ms, _ = timeit_ms(full_fn, reps=args.reps, device=device)

    # CACHE (autoregresivo)
    att_cache = clone(kv_cache=True)
    def cache_fn():
        att_cache.cache_k = None; att_cache.cache_v = None
        with torch.no_grad():
            for t in range(1, T+1):
                _ = att_cache(x[:, t-1:t, :], use_cache=True)
    cache_ms, _ = timeit_ms(cache_fn, reps=args.reps, device=device)

    # STEP (sin cache)
    att_step = clone(kv_cache=False)
    def step_fn():
        with torch.no_grad():
            for t in range(1, T+1):
                _ = att_step(x[:, :t, :], use_cache=False)
    step_ms, _ = timeit_ms(step_fn, reps=args.reps, device=device)

    # 1) Línea compacta (para run.py): full y cache en SEGUNDOS
    print(f"{full_ms/1000:.6f} {cache_ms/1000:.6f}")
    # 2) Línea humana
    print(
        f"T={T} | full={full_ms:.2f} ms | "
        f"cache={cache_ms:.2f} ms ({cache_ms/T:.3f} ms/token) | "
        f"step={step_ms:.2f} ms ({step_ms/T:.3f} ms/token) | device={device}"
    )

    # Artefactos
    (out/"bench.txt").write_text(
        f"full={full_ms/1000:.6f}s\ncache={cache_ms/1000:.6f}s\nstep={step_ms/1000:.6f}s\n",
        encoding="utf-8"
    )
    with open(out/"bench.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["T","full_ms","cache_ms","step_ms","device"])
        w.writerow([T, f"{full_ms:.4f}", f"{cache_ms:.4f}", f"{step_ms:.4f}", device.type])

    # Gráfica
    import numpy as np
    labels = ["full","cache","step (sin cache)"]; vals=[full_ms, cache_ms, step_ms]
    fig, ax = plt.subplots(figsize=(5.2,3.6))
    ax.bar(labels, vals)
    ax.set_ylabel("Latencia total (ms)")
    ax.set_title(f"Latencia (T={T})")
    for i,v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(out/"plot_latencia.png", dpi=150); plt.close(fig)

def bench_variants(args):
    """
    Compara variantes SIN cache:
      - SDPA (Multi-Head Scaled Dot-Product)   -> CausalSelfAttention(kv_cache=False)
      - MLP (multi-linear por token)           -> MLPBlock
    Mide latencia y pico de memoria (si hay GPU) para varios T.
    """
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = device_select(args.device)

    D, H  = args.d_model, args.n_heads
    Ts    = args.ctx_list

    # Modelos
    sdpa = CausalSelfAttention(D, H, kv_cache=False).to(device).eval()
    mlp  = MLPBlock(D).to(device).eval()

    rows = []  # (variant, T, mean_ms, std_ms, peak_bytes)

    for T in Ts:
        x = torch.randn(1, T, D, device=device)

        # SDPA full
        def sdpa_fn():
            with torch.no_grad():
                _ = sdpa(x)
        m_s, s_s, mem_s = time_and_peak_mem(sdpa_fn, reps=args.reps, device=device)
        rows.append(("sdpa", T, m_s, s_s, mem_s))

        # MLP full
        def mlp_fn():
            with torch.no_grad():
                _ = mlp(x)
        m_m, s_m, mem_m = time_and_peak_mem(mlp_fn, reps=args.reps, device=device)
        rows.append(("mlp", T, m_m, s_m, mem_m))

        print(f"[variants] T={T:>4} | sdpa={m_s:.2f}±{s_s:.2f} ms"
              + (f" mem={mem_s/1024/1024:.1f}MB" if mem_s is not None else "")
              + f" | mlp={m_m:.2f}±{s_m:.2f} ms"
              + (f" mem={mem_m/1024/1024:.1f}MB" if mem_m is not None else "")
              + f" | device={device}"
             )

    # Guardar CSV
    with open(out/"bench_variants.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["variant","T","mean_ms","std_ms","peak_bytes","device"])
        for v,T,m,s,mem in rows:
            w.writerow([v,T,f"{m:.4f}",f"{s:.4f}", (mem if mem is not None else ""), device.type])

    # Plot LATENCIA vs T
    import numpy as np
    Ts_sorted = sorted(set(t for _,t,_,_,_ in rows))
    def pick(vname):
        return [next(m for (v,t,m,_,_) in rows if v==vname and t==T) for T in Ts_sorted]
    sdpa_ms = pick("sdpa"); mlp_ms = pick("mlp")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(Ts_sorted, sdpa_ms, marker="o", label="SDPA (Multi-Head)")
    ax.plot(Ts_sorted, mlp_ms,  marker="o", label="MLP (multi-linear)")
    ax.set_xlabel("Contexto T")
    ax.set_ylabel("Latencia (ms)")
    ax.set_title(f"Latencia por variante (d_model={D}, heads={H})")
    ax.legend(); fig.tight_layout()
    fig.savefig(out/"plot_variants_latency.png", dpi=150); plt.close(fig)

    # Plot MEMORIA vs T (solo si GPU)
    if device.type == "cuda":
        def pick_mem(vname):
            return [next(mem for (v,t,_,_,mem) in rows if v==vname and t==T)/1024/1024 for T in Ts_sorted]
        sdpa_mem = pick_mem("sdpa"); mlp_mem = pick_mem("mlp")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(Ts_sorted, sdpa_mem, marker="o", label="SDPA (Multi-Head)")
        ax.plot(Ts_sorted, mlp_mem,  marker="o", label="MLP (multi-linear)")
        ax.set_xlabel("Contexto T"); ax.set_ylabel("Pico de memoria (MB)")
        ax.set_title("Memoria pico por variante (GPU)")
        ax.legend(); fig.tight_layout()
        fig.savefig(out/"plot_variants_memory.png", dpi=150); plt.close(fig)
    else:
        (out/"plot_variants_memory.txt").write_text(
            "Memoria pico solo disponible en GPU (CUDA).", encoding="utf-8"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--T", type=int, default=256)                 # usado en modo default
    ap.add_argument("--mode", choices=["default","variants"], default="default")
    ap.add_argument("--ctx_list", type=int, nargs="+", default=[64,128,256,384])  # usado en variants
    args = ap.parse_args()

    if args.mode == "default":
        bench_default(args)
    else:
        bench_variants(args)

if __name__ == "__main__":
    main()
