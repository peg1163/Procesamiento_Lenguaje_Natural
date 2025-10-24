# Proyecto CC02c — Mini-Transformer + atención (P2)

Entrena un mini-transformer de un bloque (decoder-only) y muestra el módulo de atención del proyecto 2 (multi-cabeza, máscara causal y caché K/V). Todo se ejecuta con `run.py`, sin bash.

---

## Dependencias
- Python 3.9+
- numpy
- torch 2.x (CPU ok)
- matplotlib

## Descripción
- **Datos**: corpus sintético reproducible (hash SHA256+SALT), splits 90/5/5.
- **Modelo**: embeddings → posición (RoPE o sinusoidal) → atención causal multi-head → MLP → logits. Se entrena a **predecir el siguiente carácter**; se reporta **perplexity** y un ejemplo.
- **P2**: atención SDPA con **máscara causal**; en inferencia usa **KV-cache**. Incluye **bench** (full vs cache) y un gráfico de latencia.
- **Salidas**: en `out/` verás `loss.png`, `plot_latencia.png`, `eval.txt`, `bench.txt`, `metrics.json`, `model.pt`, `CORPUS_SHA256.txt`.

## Uso rápido
```bash
# datos reproducibles
python run.py data && python run.py verify-corpus

# entrenar y evaluar (elige pos: rope | sinusoidal)
python run.py train --epochs 1 --pos rope
python run.py eval

# gráficos (pérdida y latencia)
python run.py plot

# bench por tamaño de contexto 
python -m src.bench --out out --T 128

# tests y paquete reproducible 
python run.py test && python run.py pack

# todo en una
python run.py all --epochs 1 --pos rope
