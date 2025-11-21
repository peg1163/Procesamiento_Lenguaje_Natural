# src/attn.py
import math
from typing import Optional
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model: int = 128, n_heads: int = 4, kv_cache: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser múltiplo de n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

  
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
 
        self.proj = nn.Linear(d_model, d_model, bias=False)


        self.register_buffer("mask", None, persistent=False)

        self.kv_cache = kv_cache
        self.cache_k: Optional[torch.Tensor] = None  # (B, H, T_total, d_head)
        self.cache_v: Optional[torch.Tensor] = None  # (B, H, T_total, d_head)

    def reset_cache(self):
        
        self.cache_k = None
        self.cache_v = None

    def _causal_mask(self, T: int) -> torch.Tensor:

        if (self.mask is None) or (self.mask.size(-1) != T):
            m = torch.tril(torch.ones(T, T, dtype=torch.bool))
            self.mask = m.view(1, 1, T, T)
        return self.mask

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:

        B, T, C = x.shape

        # Proyección y división por cabezas: (B, T, 3C) -> (3, B, H, T, d_head)
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # cada uno (B, H, T, d_head)

        # KV-cache (solo en modo autoregresivo paso a paso)
        if use_cache and self.kv_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                # concatenamos en el eje temporal
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
            k, v = self.cache_k, self.cache_v  # (B, H, T_total, d_head)

        # Scaled Dot-Product Attention (O(T^2))
        scale = 1.0 / math.sqrt(self.d_head)
        # (B, H, Tq, Tk)
        att = (q @ k.transpose(-2, -1)) * scale

        # Máscara causal
        Tq, Tk = q.size(-2), k.size(-2)
        causal = self._causal_mask(Tk)[:, :, :Tq, :Tk]
        # asegurar que la máscara está en el mismo device
        if causal.device != att.device:
            causal = causal.to(att.device)
        att = att.masked_fill(~causal, float("-inf"))

        # Softmax sobre claves (última dim)
        att = torch.softmax(att, dim=-1)

        # Ponderar valores y recombinar cabezas
        y = att @ v                                # (B, H, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        return self.proj(y)
