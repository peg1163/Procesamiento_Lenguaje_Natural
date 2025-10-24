import unittest
import torch
from src.attn import CausalSelfAttention

class TestKVCache(unittest.TestCase):
    def test_step_vs_full_shapes_equal(self):
        D, H, B, T = 16, 2, 1, 12
        x = torch.randn(B, T, D)

        a_full = CausalSelfAttention(D, H, kv_cache=False)
        a_step = CausalSelfAttention(D, H, kv_cache=True)

        with torch.no_grad():
            y_full = a_full(x)  # (B,T,D)

            ys = []
            for t in range(T):
                ys.append(a_step(x[:, :t+1, :], use_cache=True)[:, -1:, :])
            y_step = torch.cat(ys, dim=1)

        # Mismo tamaño y finitud 
        self.assertEqual(y_full.shape, y_step.shape)
        self.assertFalse(torch.isnan(y_step).any().item())

if __name__ == "__main__":
    unittest.main()
