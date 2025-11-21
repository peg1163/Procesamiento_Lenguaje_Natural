import torch

from src.models.posenc import (
    SinusoidalPositionalEncoding,
    apply_rope,
    build_alibi_bias,
)


def test_sinusoidal_shape():
    d_model = 64
    max_len = 100
    pe = SinusoidalPositionalEncoding(d_model, max_len)
    x = torch.zeros(2, 10, d_model)
    out = pe(x)
    assert out.shape == x.shape


def test_rope_preserves_norm():
    B, T, H, Dh = 2, 16, 4, 8
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    positions = torch.arange(T)

    q2, k2 = apply_rope(q, k, positions)
    # norma por vector debe ser similar
    assert torch.allclose(
        q.norm(dim=-1),
        q2.norm(dim=-1),
        atol=1e-5,
        rtol=1e-3,
    )


def test_alibi_bias_monotonic():
    n_heads = 4
    T = 8
    bias = build_alibi_bias(n_heads, T)  # [n_heads,T,T]
    assert bias.shape == (n_heads, T, T)

    # para una cabeza, la penalización aumenta con la distancia
    h = 0
    for i in range(T):
        for j in range(T - 1):
            if j < i and j + 1 < i:
                assert bias[h, i, j] >= bias[h, i, j + 1]
