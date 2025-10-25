
import argparse, os, sys, io, json, random, hashlib, tarfile, time, shutil
from pathlib import Path
import matplotlib.pyplot as plt
import subprocess, statistics as stats
from trace import Trace
import importlib.util, runpy
import  unittest
import torch, tracemalloc, gc, csv
from src.attn import CausalSelfAttention
import numpy as np
import pathlib
from trace import Trace
OUT = Path(os.getenv("OUT", "out"))
DIST = Path(os.getenv("DIST", "dist"))
SEED = int(os.getenv("SEED", "1337"))
SALT = os.getenv("SALT", "CC02")
SOURCE_DATE_EPOCH = int(os.getenv("SOURCE_DATE_EPOCH", str(int(time.time()))))

def info(msg): print(f"[+] {msg}")
def warn(msg): print(f"[!] {msg}")
def fail(msg): print(f"[x] {msg}"); sys.exit(1)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sha256_bytes(b: bytes, salt: str = "") -> str:
    return hashlib.sha256(b + salt.encode("utf-8")).hexdigest()

def py(cmd: list[str], check=True):
    
    full = [sys.executable] + cmd
    info(" ".join(full))
    return subprocess.run(full, check=check)

def gen_corpus(out: Path = OUT, seed: int = SEED, salt: str = SALT):
    ensure_dir(out)
    raw = out / "corpus_raw.txt"
    meta = out / "corpus_meta.json"
    sumf = out / "CORPUS_SHA256.txt"

    rnd = random.Random(seed)
    lines = []

    for _ in range(2000):
        k = rnd.randint(5, 40)
        lines.append("".join(rnd.choice("abc()") for _ in range(k)))

    for _ in range(2000):
        a = rnd.randint(0, 9999)
        b = rnd.randint(0, 9999)
        lines.append(f"{a}+{b}=")

    with io.open(raw, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    h = sha256_bytes(raw.read_bytes(), salt)
    sumf.write_text(h + "\n", encoding="utf-8", newline="\n")
    meta.write_text(json.dumps({"seed": seed, "salt": salt, "sha256": h, "n_lines": len(lines)}, indent=2),
                    encoding="utf-8", newline="\n")
    info(f"Corpus generado en {out} (sha256={h[:12]}…)")

def verify_corpus(out: Path = OUT, seed: int = SEED, salt: str = SALT):
    sumf = out / "CORPUS_SHA256.txt"
    if not sumf.exists():
        fail("No existe out/CORPUS_SHA256.txt. Ejecuta primero: python run.py data")
    expected = sumf.read_text(encoding="utf-8").strip()
    tmp = out / "_tmp_verify"
    if tmp.exists(): shutil.rmtree(tmp)
    ensure_dir(tmp)
    gen_corpus(tmp, seed, salt)
    h2 = (tmp / "CORPUS_SHA256.txt").read_text(encoding="utf-8").strip()
    shutil.rmtree(tmp)

    if expected == h2:
        info("VERIFICADO(hash coincide)")
    else:
        warn(f"esperado: {expected}")
        warn(f"obtenido: {h2}")
        fail("NO COINCIDE")

def step_train(args):
    ensure_dir(OUT)
    cmd = ["-m", "src.train", "--out", str(OUT),
           "--epochs", str(args.epochs),
           "--lr", str(args.lr),
           "--batch", str(args.batch),
           "--ctx", str(args.ctx),
           "--pos", args.pos,
           "--device", args.device]
    py(cmd)

def step_eval(_args):
    ensure_dir(OUT)
    py(["-m", "src.eval", "--out", str(OUT)])

def step_bench(_args):
    ensure_dir(OUT)
    py(["-m", "src.bench", "--out", str(OUT)])

def step_plot(_args):
    
    m = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    plt.figure()
    plt.plot(m["train"]["step"], m["train"]["loss"])
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("Train Loss")
    plt.savefig(OUT / "loss.png")
    print(f"[plot] guardado {OUT/'loss.png'}")

    def run_bench_once(T: int):
        p = subprocess.run([sys.executable, "-m", "src.bench", "--out", str(OUT), "--T", str(T)],
                           capture_output=True, text=True, check=True)
        line = p.stdout.strip() or (OUT / "bench.txt").read_text(encoding="utf-8").strip()
        toks = line.replace("full","").replace("cache","").split()
        full_s = float(toks[0]); cache_s = float(toks[1])
        return full_s*1000.0, cache_s*1000.0  # ms

    Ts = [128, 256]
    reps = 3
    means_full = []; sds_full = []
    means_cache = []; sds_cache = []
    for T in Ts:
        full_ms = []; cache_ms = []
        for _ in range(reps):
            fms, cms = run_bench_once(T)
            full_ms.append(fms); cache_ms.append(cms)
        means_full.append(stats.mean(full_ms));   sds_full.append(stats.pstdev(full_ms))
        means_cache.append(stats.mean(cache_ms)); sds_cache.append(stats.pstdev(cache_ms))

    plt.figure(figsize=(6,4))

    x = np.arange(len(Ts)); width = 0.35
    plt.bar(x - width/2, means_full,  width, yerr=sds_full,  capsize=4, label="full")
    plt.bar(x + width/2, means_cache, width, yerr=sds_cache, capsize=4, label="cache")
    plt.xticks(x, [str(t) for t in Ts]); plt.xlabel("Contexto T"); plt.ylabel("Latencia (ms)")
    plt.title("Latencia: full (un forward) vs cache (autoregresivo)")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "plot_latencia.png")
    print(f"[plot] guardado {OUT/'plot_latencia.png'}")


    def peak_bytes_cpu(fn):
        gc.collect()
        tracemalloc.start()
        fn()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return int(peak)

    def peak_bytes_cuda(fn):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())

    use_cuda = torch.cuda.is_available()
    peak_fn = peak_bytes_cuda if use_cuda else peak_bytes_cpu

    def mem_full_once(T: int, D=128, H=4):
        attn = CausalSelfAttention(D, H, kv_cache=False).eval()
        x = torch.randn(1, T, D, device=("cuda" if use_cuda else "cpu"))
        with torch.no_grad():
            def _f(): attn(x)
            return peak_fn(_f)

    def mem_cache_once(T: int, D=128, H=4):
        attn = CausalSelfAttention(D, H, kv_cache=True).eval()
        x = torch.randn(1, T, D, device=("cuda" if use_cuda else "cpu"))
        with torch.no_grad():
            def _f():
                y = []
                for t in range(T):
                    y.append(attn(x[:, t:t+1, :], use_cache=True))
            return peak_fn(_f)

    Ts_mem = [64, 128, 256]
    mem_rows = [("T","mode","peak_bytes")]
    mem_full = []; mem_cache = []
    for T in Ts_mem:
        b_full = mem_full_once(T)
        b_cache = mem_cache_once(T)
        mem_rows.append((T,"full", b_full))
        mem_rows.append((T,"cache",b_cache))
        mem_full.append(b_full/1e6)
        mem_cache.append(b_cache/1e6)

    with open(OUT/"mem.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerows(mem_rows)
    print(f"[plot] guardado {OUT/'mem.csv'}")

    plt.figure(figsize=(6,4))
    plt.plot(Ts_mem, mem_full, marker="o", label="full")
    plt.plot(Ts_mem, mem_cache, marker="o", label="cache")
    plt.xlabel("Contexto T"); plt.ylabel("Memoria pico (MB)")
    plt.title("Memoria pico vs contexto (aprox.)")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "plot_memoria.png")
    print(f"[plot] guardado {OUT/'plot_memoria.png'}")


def step_test(_args):
    
    PKG = pathlib.Path("src")
    EXCLUDE = set()  
    files = [p for p in PKG.rglob("*.py") if p.name not in EXCLUDE]

    def run_unittests():
        suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)

    
    tr = Trace(count=True, trace=False, ignoremods=["torch", "numpy", "matplotlib"])
    tr.runfunc(run_unittests)

    
    counts = tr.results().counts  # dict[(filename, lineno) -> hits]
    counted = {(os.path.abspath(f), ln) for (f, ln) in counts.keys()}

    total = 0
    hit = 0
    for f in files:
        abspath = os.path.abspath(str(f))
        with open(f, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                total += 1
                if (abspath, i) in counted:
                    hit += 1

    ratio = (hit / total) if total else 1.0
    print("\nOK")
    print(f"[+] Coverage (aprox): {ratio:.2%}")
    if ratio + 1e-9 < 0.70:
        print("[x] Cobertura por debajo de 70%")
        sys.exit(1)



def step_pack(_args):
    ensure_dir(OUT); ensure_dir(DIST)
    tar_path = DIST / "proyecto-cc02.tar.gz"

    # Lista de archivos a incluir
    def include(p: Path) -> bool:
        name = p.name
        if name.endswith(".pyc") or name == "__pycache__":
            return False
        return True

    files = []
    for root, dirs, fnames in os.walk("."):
        # excluir __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in fnames:
            p = Path(root) / fn
            if include(p):
                files.append(p)
    files = sorted(files, key=lambda x: str(x).replace("\\", "/"))

    
    info(f"Empaquetando {len(files)} archivos → {tar_path}")
    with tarfile.open(tar_path, "w:gz", format=tarfile.GNU_FORMAT) as tar:
        for p in files:
            ti = tarfile.TarInfo(str(p).replace("\\", "/"))
            st = p.stat()
            ti.size = st.st_size
            ti.mtime = SOURCE_DATE_EPOCH  
            ti.uid = 0; ti.gid = 0; ti.uname = ""; ti.gname = ""
            ti.mode = 0o644
            with open(p, "rb") as f:
                tar.addfile(ti, f)
    h = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (OUT / "PACKAGE_SHA256.txt").write_text(h + "\n", encoding="utf-8", newline="\n")
    info(f"SHA256 paquete: {h[:12]}…")


def step_all(args):
    gen_corpus(OUT, SEED, SALT)
    verify_corpus(OUT, SEED, SALT)
    step_train(args)
    step_eval(args)
    step_bench(args)
    step_plot(args)
    step_test(args)
    step_pack(args)
    info("Pipeline COMPLETO")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("data", help="Genera el corpus (SEED+SALT)")
    sub.add_parser("verify-corpus", help="Regenera el corpus y compara SHA")
    ptrain = sub.add_parser("train", help="Entrena Mini-Transformer")
    ptrain.add_argument("--epochs", type=int, default=1)
    ptrain.add_argument("--lr", type=float, default=3e-4)
    ptrain.add_argument("--batch", type=int, default=32)
    ptrain.add_argument("--ctx", type=int, default=128)
    ptrain.add_argument("--pos", choices=["rope", "sinusoidal"], default="rope")
    ptrain.add_argument("--device", default="auto")

    sub.add_parser("eval", help="Evalúa perplexity y ejemplo")
    sub.add_parser("bench", help="Benchmark SDPA vs KV-cache")
    sub.add_parser("plot", help="Genera out/loss.png")
    sub.add_parser("test", help="Tests + cobertura mínima")
    sub.add_parser("pack", help="Empaqueta determinista en dist/")

    pall = sub.add_parser("all", help="Ejecuta todo el pipeline")
    pall.add_argument("--epochs", type=int, default=1)
    pall.add_argument("--lr", type=float, default=3e-4)
    pall.add_argument("--batch", type=int, default=32)
    pall.add_argument("--ctx", type=int, default=128)
    pall.add_argument("--pos", choices=["rope", "sinusoidal"], default="rope")
    pall.add_argument("--device", default="auto")

    args = ap.parse_args()

    if args.cmd == "data":
        gen_corpus(OUT, SEED, SALT)
    elif args.cmd == "verify-corpus":
        verify_corpus(OUT, SEED, SALT)
    elif args.cmd == "train":
        step_train(args)
    elif args.cmd == "eval":
        step_eval(args)
    elif args.cmd == "bench":
        step_bench(args)
    elif args.cmd == "plot":
        step_plot(args)
    elif args.cmd == "test":
        step_test(args)
    elif args.cmd == "pack":
        step_pack(args)
    elif args.cmd == "all":
        step_all(args)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
