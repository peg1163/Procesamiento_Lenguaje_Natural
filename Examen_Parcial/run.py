
import argparse, os, sys, io, json, random, hashlib, tarfile, time, shutil, subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import sys, subprocess, statistics as stats
from trace import Trace
import importlib.util, runpy, pathlib
# ----- Defaults / rutas -----
OUT = Path(os.getenv("OUT", "out"))
DIST = Path(os.getenv("DIST", "dist"))
SEED = int(os.getenv("SEED", "1337"))
SALT = os.getenv("SALT", "CC02")
SOURCE_DATE_EPOCH = int(os.getenv("SOURCE_DATE_EPOCH", str(int(time.time()))))

# ============ UTILIDADES ============

def info(msg): print(f"[+] {msg}")
def warn(msg): print(f"[!] {msg}")
def fail(msg): print(f"[x] {msg}"); sys.exit(1)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sha256_bytes(b: bytes, salt: str = "") -> str:
    return hashlib.sha256(b + salt.encode("utf-8")).hexdigest()

def py(cmd: list[str], check=True):
    """Ejecuta un módulo Python como subproceso (para -m src.*)."""
    full = [sys.executable] + cmd
    info(" ".join(full))
    return subprocess.run(full, check=check)

# ============ DATASET: generar / verificar ============

def gen_corpus(out: Path = OUT, seed: int = SEED, salt: str = SALT):
    ensure_dir(out)
    raw = out / "corpus_raw.txt"
    meta = out / "corpus_meta.json"
    sumf = out / "CORPUS_SHA256.txt"

    rnd = random.Random(seed)
    lines = []

    # 1) Patrones con () y abc
    for _ in range(2000):
        k = rnd.randint(5, 40)
        lines.append("".join(rnd.choice("abc()") for _ in range(k)))

    # 2) Sumas con acarreo: "A+B="
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

    # Regeneramos temporalmente y comparamos
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

# ============ TRAIN / EVAL / BENCH / PLOT ============
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
    # 1) Curva de pérdida (igual que antes)

    m = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    plt.figure()
    plt.plot(m["train"]["step"], m["train"]["loss"])
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("Train Loss")
    plt.savefig(OUT / "loss.png")
    print(f"Plot guardado en {OUT/'loss.png'}")

    # 2) MUY SIMPLE: correr bench y generar una gráfica de latencia
    #    (T=128 y 256, 3 repeticiones cada uno), y guardar out/plot_latencia.png

    def run_bench_once(T: int):
        # Ejecuta: python -m src.bench --out out --T T
        # src.bench imprime "full X cache Y" en stdout y escribe out/bench.txt
        p = subprocess.run([sys.executable, "-m", "src.bench", "--out", str(OUT), "--T", str(T)],
                           capture_output=True, text=True, check=True)
        line = p.stdout.strip()
        if not line or "full" not in line:
            line = (OUT / "bench.txt").read_text(encoding="utf-8").strip()
        # parseo rápido
        toks = line.replace("full","").replace("cache","").split()
        full_s = float(toks[0]); cache_s = float(toks[1])
        return full_s*1000.0, cache_s*1000.0  # a ms

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

    # gráfico bar chart muy simple
    
    x = range(len(Ts)); width = 0.35
    plt.figure(figsize=(6,4))
    plt.bar([i - width/2 for i in x], means_full,  width, yerr=sds_full,  capsize=4, label="full")
    plt.bar([i + width/2 for i in x], means_cache, width, yerr=sds_cache, capsize=4, label="cache")
    plt.xticks(list(x), [str(t) for t in Ts])
    plt.xlabel("Contexto T")
    plt.ylabel("Latencia (ms)")
    plt.title("Latencia: full (un forward) vs cache (autoregresivo)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "plot_latencia.png")
    print(f"Plot de latencia guardado en {OUT/'plot_latencia.png'}")


# ============ TESTS + COBERTURA (stdlib) ============
def step_test(_args):
    # Ejecuta unittest discover y luego una cobertura aproximada con trace
    info("Ejecutando tests…")
    res = py(["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"], check=False)
    if res.returncode != 0:
        fail("Tests fallaron")

    # Cobertura aproximada


    pkg = Path("src")
    exclude = {"mini_transformer.py", "train.py", "bench.py", "eval.py"}  # exentos (numéricos)
    files = [p for p in pkg.rglob("*.py") if p.name not in exclude]

    tr = Trace(count=True, trace=False, ignoremods=["torch", "numpy", "matplotlib"])
    for t in Path("tests").rglob("test_*.py"):
        runpy.run_path(str(t))

    counts = tr.results().counts
    total = 0; hit = 0
    for f in files:
        src = f.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(src, 1):
            if line.strip().startswith("#"): 
                continue
            total += 1
            if (str(f), i) in counts:
                hit += 1
    ratio = (hit / total) if total else 1.0
    info(f"Coverage (aprox): {ratio:.2%}")
    if ratio < 0.70:
        fail("Cobertura por debajo de 70%")

# ============ PACK reproducible ============
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

    # Empaquetado determinista
    info(f"Empaquetando {len(files)} archivos → {tar_path}")
    with tarfile.open(tar_path, "w:gz", format=tarfile.GNU_FORMAT) as tar:
        for p in files:
            ti = tarfile.TarInfo(str(p).replace("\\", "/"))
            st = p.stat()
            ti.size = st.st_size
            ti.mtime = SOURCE_DATE_EPOCH  # fijamos mtime
            ti.uid = 0; ti.gid = 0; ti.uname = ""; ti.gname = ""
            ti.mode = 0o644
            with open(p, "rb") as f:
                tar.addfile(ti, f)
    h = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (OUT / "PACKAGE_SHA256.txt").write_text(h + "\n", encoding="utf-8", newline="\n")
    info(f"SHA256 paquete: {h[:12]}…")

# ============ ALL ============
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

# ============ CLI ============
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # data
    sub.add_parser("data", help="Genera el corpus (SEED+SALT)")
    sub.add_parser("verify-corpus", help="Regenera el corpus y compara SHA")

    # train/eval/bench/plot/test/pack/all
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
