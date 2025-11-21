import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):


    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    
        T = x.size(1)
        pos_emb = self.pe[:, start_pos : start_pos + T, :]  # [1,T,D]
        return x + pos_emb


# ========= RoPE ========= #

def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    B, H, T, D = q.shape
    device = q.device

    if positions is None:
        positions = torch.arange(T, device=device)
    positions = positions.to(device)

    # Frecuencias para cada par (even, odd)
    dim = torch.arange(0, D, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (dim / D))  # [D/2]

    # [T, D/2]
    freqs = torch.einsum("t,d->td", positions.float(), inv_freq)
    cos = freqs.cos()[None, None, :, :]  # [1,1,T,D/2]
    sin = freqs.sin()[None, None, :, :]

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        x_out = torch.zeros_like(x)
        x_out[..., 0::2] = x_rot_even
        x_out[..., 1::2] = x_rot_odd
        return x_out

    return _rotate(q), _rotate(k)


# ========= ALiBi ========= #

def _get_alibi_slopes(n_heads: int) -> list[float]:


    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2.0 ** (-2.0 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)

    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    slopes = get_slopes_power_of_2(closest_power_of_2)
    extra = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
    slopes = slopes + extra[: n_heads - closest_power_of_2]
    return slopes


def build_alibi_bias(
    n_heads: int,
    max_len: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Construye el bias ALiBi:

    return: [n_heads, max_len, max_len], donde para cada cabeza h:
        bias[h, i, j] = slope_h * (i - j)
    Es monótono en j para j < i (lo que usamos en el test).
    """
    device = device or torch.device("cpu")
    slopes = torch.tensor(_get_alibi_slopes(n_heads), device=device, dtype=dtype)
    slopes = slopes.view(n_heads, 1, 1)  # [H,1,1]

    pos = torch.arange(max_len, device=device)
    # rel_pos[i, j] = i - j
    rel_pos = pos.view(max_len, 1) - pos.view(1, max_len)  # [T,T]
    rel_pos = rel_pos.view(1, max_len, max_len)  # [1,T,T]

    bias = slopes * rel_pos  # [H,T,T]
    return bias
