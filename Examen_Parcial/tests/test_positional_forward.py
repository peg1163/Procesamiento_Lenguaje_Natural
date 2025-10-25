import unittest, torch
from src.mini_transformer import MiniTransformer

class TestPositionalForward(unittest.TestCase):
    def test_rope_and_sinusoidal_forward_shapes(self):
        torch.manual_seed(123)
        V = 32
        m_rope = MiniTransformer(vocab_size=V, d_model=64, n_heads=4, d_mlp=128, pos="rope")
        m_sin  = MiniTransformer(vocab_size=V, d_model=64, n_heads=4, d_mlp=128, pos="sinusoidal")
        x = torch.randint(0, V, (2, 16))
        y1 = m_rope(x); y2 = m_sin(x)
        self.assertEqual(y1.shape, (2,16,V))
        self.assertEqual(y2.shape, (2,16,V))
