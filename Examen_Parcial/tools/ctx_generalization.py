import argparse, sys, runpy, re, json
from pathlib import Path
from unittest.mock import patch

def run_eval_with_ctx(out, ctx):
    
    with patch.object(sys, "argv", ["src.eval", "--out", str(out), "--ctx", str(ctx)]):
        runpy.run_module("src.eval", run_name="__main__", alter_sys=True)
    txt = (out/"eval.txt").read_text(encoding="utf-8")
    m = re.search(r"perplexity[: ]+([0-9.]+)", txt, re.I)
    return float(m.group(1)) if m else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--ctx_list", nargs="+", type=int, default=[128, 256, 384])
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = [("ctx","perplexity")]
    for ctx in args.ctx_list:
        ppl = run_eval_with_ctx(out, ctx)
        rows.append((ctx, ppl))
        print(f"[generalization] ctx={ctx} -> ppl={ppl:.3f}")


    csv = out/"ctx_generalization.csv"
    with open(csv, "w", encoding="utf-8") as f:
        f.write("ctx,perplexity\n")
        for ctx, ppl in rows[1:]:
            f.write(f"{ctx},{ppl}\n")
    print(f"[generalization] guardado {csv}")

    
    (out/"perplexity.json").write_text(json.dumps({"last": rows[-1][1]}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
