import math
from typing import Optional
import torch
import torch.nn as nn


def make_causal_mask(
    T: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.uint8,
) -> torch.Tensor:
  
    device = device or torch.device("cpu")
    mask = torch.tril(torch.ones((T, T), dtype=dtype, device=device))
    return mask.view(1, 1, T, T)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k, v: [B, H, T, Dh]
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,T,T]

        if attn_bias is not None:
            scores = scores + attn_bias

        if attn_mask is not None:
            # attn_mask == 0 → -inf (apagado)
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B,H,T,Dh]
        return out, attn
