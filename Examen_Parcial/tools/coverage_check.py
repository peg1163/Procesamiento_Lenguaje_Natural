import argparse, runpy, sys, pathlib
from trace import Trace

parser = argparse.ArgumentParser()
parser.add_argument('--pkg', default='src')
parser.add_argument('--min', type=float, default=0.7)
parser.add_argument('--exclude', default='')  # coma-separado por nombre de archivo
args = parser.parse_args()

pkg = pathlib.Path(args.pkg)
excl = set(x.strip() for x in args.exclude.split(',') if x.strip())


files = [p for p in pkg.rglob('*.py') if p.name not in excl]


tr = Trace(count=True, trace=False, ignoremods=['torch','numpy','matplotlib'])

def _run_file(path_str: str):
    runpy.run_path(path_str, run_name="__main__")


for t in pathlib.Path('tests').rglob('test_*.py'):
    tr.runfunc(_run_file, str(t))

counts = tr.results().counts 

total = 0
hit = 0
for f in files:
    src_lines = f.read_text(encoding='utf-8').splitlines()
    for i, line in enumerate(src_lines, 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        total += 1
        if (str(f), i) in counts:
            hit += 1

ratio = (hit / total) if total else 1.0
print(f"Coverage (approx): {ratio:.2%} (min={args.min:.0%})")
if ratio + 1e-9 < args.min:
    print("FAIL: coverage below minimum")
    sys.exit(1)
