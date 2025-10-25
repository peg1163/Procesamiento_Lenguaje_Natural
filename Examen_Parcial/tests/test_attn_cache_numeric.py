import unittest, torch
from src.attn import CausalSelfAttention

class TestKVCacheNumeric(unittest.TestCase):
    def test_kv_cache_matches_full_last_token(self):
        torch.manual_seed(42)
        B, T, D, H = 1, 1, 64, 4   # T=1 evita desajustes posicionales
        x = torch.randn(B, T, D)

        att_full = CausalSelfAttention(D, H, kv_cache=False).eval()
        att_inc  = CausalSelfAttention(D, H, kv_cache=True).eval()
        att_inc.load_state_dict(att_full.state_dict())

        with torch.no_grad():
            y_full = att_full(x)            # (B,1,D)
            ys = []
            for t in range(T):              # T=1 ⇒ un solo paso
                ys.append(att_inc(x[:, t:t+1, :], use_cache=True))
            y_inc = torch.cat(ys, 1)

        self.assertTrue(torch.allclose(y_inc, y_full, atol=1e-5, rtol=1e-5))
