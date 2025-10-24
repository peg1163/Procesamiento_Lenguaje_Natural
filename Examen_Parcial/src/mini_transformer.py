import math
import torch
import torch.nn as nn
from .attn import CausalSelfAttention

class Sinusoidal(nn.Module):
    """Posicional sinusoidal (estática)."""
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor, start: int = 0) -> torch.Tensor:
        # x: (B,T,C)
        pe = self.pe[start:start + x.size(1)].to(dtype=x.dtype, device=x.device)
        return x + pe.unsqueeze(0)

class RoPE(nn.Module):
    """Rotary Position Embedding aplicado sobre el embedding de entrada (simple)."""
    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 2 == 0, "RoPE requiere dimensión par"
        self.d_model = d_model
        self.half = d_model // 2  # d/2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,C)  -> devuelve (B,T,C)
        Construimos theta con forma exacta (B,T,d/2) para evitar ambigüedades de broadcasting.
        """
        B, T, C = x.shape
        d2 = self.half
        # Frecuencias: (d/2,)
        freqs = torch.exp(
            torch.arange(0, d2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / d2)
        )  # (d/2,)

        # Posiciones: (T,)
        pos = torch.arange(T, device=x.device, dtype=x.dtype)  # (T,)
        # Theta: (T, d/2) -> (1,T,d/2) -> (B,T,d/2)
        theta = (pos[:, None] * freqs[None, :]).unsqueeze(0).expand(B, T, d2)

        # Separar pares (B,T,d/2)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        xr1 = x1 * cos_t - x2 * sin_t
        xr2 = x1 * sin_t + x2 * cos_t
        # Intercalar de nuevo a (B,T,C)
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = xr1
        x_rot[..., 1::2] = xr2
        return x_rot

class Block(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, d_mlp: int = 256, pos: str = "rope"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.pos_kind = pos
        self.pos_sin = Sinusoidal(d_model)
        self.pos_rope = RoPE(d_model)

    def add_pos(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_rope(x) if self.pos_kind == "rope" else self.pos_sin(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_pos(x)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniTransformer(nn.Module):
    """Decoder-only minimal: token embedding → 1 bloque → LN → cabeza vocab."""
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, d_mlp: int = 256, pos: str = "rope"):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.block = Block(d_model, n_heads, d_mlp, pos)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)      # (B,T,C)
        x = self.block(x)          # (B,T,C)
        x = self.ln(x)             # (B,T,C)
        return self.head(x)        # (B,T,V)
