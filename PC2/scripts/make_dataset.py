
import argparse
import csv
import random
from pathlib import Path
from typing import Tuple

import numpy as np

SCRIPT_DIR   = Path(__file__).resolve().parent          
PROJECT_ROOT = SCRIPT_DIR.parent                        
DATA_DIR     = PROJECT_ROOT / "data"                    

CATEGORIES = ["Positivo", "Negativo", "Neutral"]

POS = [
    "La cámara del celular es espectacular",
    "La batería dura todo el día",
    "La calidad de construcción es excelente",
    "El teclado responde muy bien",
]
NEG = [
    "El software es confuso y se bloquea",
    "La batería se descarga muy rápido",
    "La pantalla llegó con píxeles muertos",
    "El envío fue muy lento",
]
NEU = [
    "El diseño es moderno pero el precio es alto",
    "El empaque llegó en buenas condiciones",
    "La garantía es de un año",
    "La tienda está en línea y física",
]

def synth_review(rng: random.Random) -> Tuple[str, str]:
    """Devuelve (texto, categoría) de forma determinista usando rng."""
    cat = rng.choice(CATEGORIES)
    base = {"Positivo": POS, "Negativo": NEG, "Neutral": NEU}[cat]
    s1 = rng.choice(base)
    s2 = rng.choice([".", "!", "..."])
    extras = rng.choice([
        "", " El color me gustó", " El precio es razonable", " El manual no ayuda mucho",
        " El envío fue regular", " El rendimiento es aceptable"
    ])
    return f"{s1}{s2}{extras}", cat

def run_make_dataset(seed: int, n: int, filename: str = "nlp_prueba_cc0c2.csv") -> Path:

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / filename

    rng = random.Random(seed)
    np.random.seed(seed)

    rows = [synth_review(rng) for _ in range(n)]
    rng.shuffle(rows)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["Texto", "Categoría"])
        w.writerows(rows)

    print(f"[make_dataset] Escrito: {out_path}")
    print(f"[make_dataset] Filas: {len(rows)}")
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Genera dataset sintético reproducible en ../data/<archivo>")
    ap.add_argument("--seed", type=int, default=42, help="Semilla determinista")
    ap.add_argument("--n", type=int, default=5000, help="Total de oraciones a generar")
    ap.add_argument("--filename", type=str, default="nlp_prueba_cc0c2.csv",
                    help="Nombre del archivo dentro de la carpeta data/")
    args = ap.parse_args()
    run_make_dataset(seed=args.seed, n=args.n, filename=args.filename)

if __name__ == "__main__":
    main()
