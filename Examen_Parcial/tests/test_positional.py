import unittest
import torch
from src.mini_transformer import Sinusoidal, RoPE

class TestPositional(unittest.TestCase):
    def test_shapes(self):
        x = torch.zeros(2, 8, 16)  # (B,T,C)
        self.assertEqual(Sinusoidal(16)(x).shape, x.shape)
        self.assertEqual(RoPE(16)(x).shape, x.shape)

    def test_rope_requires_even_dim(self):
        with self.assertRaises(AssertionError):
            _ = RoPE(15)

if __name__ == "__main__":
    unittest.main()
