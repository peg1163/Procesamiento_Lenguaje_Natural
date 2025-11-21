import torch

from src.models.attention import make_causal_mask


def test_causal_mask_shape_and_values():
    T = 5
    mask = make_causal_mask(T)  # se espera shape [1,1,T,T]
    assert mask.shape == (1, 1, T, T)

    # diagonal e inferior = 1, superior = 0
    for i in range(T):
        for j in range(T):
            val = mask[0, 0, i, j].item()
            if j <= i:
                assert val == 1, f"mask[{i},{j}] debería ser 1"
            else:
                assert val == 0, f"mask[{i},{j}] debería ser 0"
