import unittest
import torch
from src.attn import CausalSelfAttention

class TestMask(unittest.TestCase):
    def test_causal_output_shape(self):
        att = CausalSelfAttention(d_model=8, n_heads=2)
        x = torch.zeros(1, 4, 8)          # (B,T,C)
        y = att(x)                         # forward sin cache
        self.assertEqual(y.shape, (1, 4, 8))

    def test_no_nan_after_mask(self):
        att = CausalSelfAttention(d_model=16, n_heads=4)
        x = torch.randn(2, 5, 16)
        y = att(x)
        self.assertFalse(torch.isnan(y).any().item())

if __name__ == "__main__":
    unittest.main()
