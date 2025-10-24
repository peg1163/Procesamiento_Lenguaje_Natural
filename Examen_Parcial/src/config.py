from dataclasses import dataclass
@dataclass
class Config:
    vocab: str = "0123456789+()=abc\n" # caracteres de la tarea
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 256
    n_layers: int = 1 # Mini-Transformer: 1 bloque
    ctx_len: int = 128
    dropout: float = 0.0
    pos_encoding: str = "rope" # rope | sinusoidal