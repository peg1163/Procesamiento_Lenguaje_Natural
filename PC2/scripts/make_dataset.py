import argparse
import random
import pandas as pd

parser = argparse.ArgumentParser(description="Generar corpus de reseñas")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n-samples", type=int, default=5000)
parser.add_argument("--out", type=str, default="data/nlp_prueba_cc0c2.csv")
parser.add_argument("--balance", action="store_true")
args = parser.parse_args()

random.seed(args.seed)
productos = ["celular", "laptop", "auriculares"]
adj_pos = ["excelente", "rápido", "increíble"]
adj_neg = ["defectuoso", "lento", "caro"]
adj_neu = ["normal", "estándar", "promedio"]
data = []
for _ in range(args.n_samples):
    prod = random.choice(productos)
    cat = random.choice(["Positivo", "Negativo", "Neutral"])
    adj = random.choice(adj_pos if cat == "Positivo" else adj_neg if cat == "Negativo" else adj_neu)
    texto = f"El {prod} es {adj}"
    data.append([texto, cat])
pd.DataFrame(data, columns=["Texto", "Categoría"]).to_csv(args.out, index=False)