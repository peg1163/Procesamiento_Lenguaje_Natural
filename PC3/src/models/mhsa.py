from typing import Optional

import torch
import torch.nn as nn

from .attention import ScaledDotProductAttention, make_causal_mask
from .posenc import apply_rope


class MultiHeadSelfAttention(nn.Module):
    #Multi-Head Self-Attention con soporte para:
    #Máscara causal
    #RoPE 
    #Bias ALiBi 
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser divisible por n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        alibi_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        #x: [B, T, D]
        #attn_mask: [1,1,T,T] o [B,1,T,T]
        #positions: [T] (para RoPE)
        #alibi_bias: [H, T, T] (bias por cabeza)
       
        B, T, C = x.shape
        H, Dh = self.n_heads, self.d_head

        # Proyecciones
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        # RoPE (rota q y k)
        if self.use_rope and positions is not None:
            q, k = apply_rope(q, k, positions)

        # Máscara causal por defecto
        if attn_mask is None:
            attn_mask = make_causal_mask(T, device=x.device)  # [1,1,T,T]

        # ALiBi: expandimos a [1,H,T,T] para que se broadcast con B
        attn_bias = None
        if alibi_bias is not None:
            attn_bias = alibi_bias.unsqueeze(0)  # [1,H,T,T]

        # Atención
        attn_out, _ = self.attn(q, k, v, attn_mask=attn_mask, attn_bias=attn_bias)

        # Volver a [B,T,D]
        out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out
