import math
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, kv_cache: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser múltiplo de n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.kv_cache = kv_cache
        self.register_buffer("mask", None, persistent=False)
        self.cache_k = None  # (B,H,T,d)
        self.cache_v = None  # (B,H,T,d)

    def _causal_mask(self, T: int) -> torch.Tensor:
        # (1,1,T,T) booleana: permite i>=j
        if (self.mask is None) or (self.mask.size(-1) != T):
            m = torch.tril(torch.ones(T, T, dtype=torch.bool))
            self.mask = m.view(1, 1, T, T)
        return self.mask

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        ## se le ingresa un  (B,T,C)
        ## y retorna  (B,T,C)
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,T,d)
        if use_cache and self.kv_cache:
            # Concatena en el eje temporal (T)
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
            k, v = self.cache_k, self.cache_v  # (B,H,T_total,d)

        scale = 1.0 / math.sqrt(self.d_head)
        att = (q @ k.transpose(-2, -1)) * scale  # (B,H,Tq,Tk)

        Tq, Tk = q.size(-2), k.size(-2)
        causal = self._causal_mask(Tk)[:, :, :Tq, :Tk]
        att = att.masked_fill(~causal, float("-inf"))

        att = torch.softmax(att, dim=-1)
        y = att @ v  # (B,H,T,d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
