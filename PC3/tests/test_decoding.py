import torch

from src.decoding import sample_top_k_top_p


def test_sample_top_k_reduces_support():
    logits = torch.ones(1, 100)  # probs uniformes
    top_k = 10
    token = sample_top_k_top_p(logits, top_k=top_k, top_p=1.0, temperature=1.0)
    assert token.shape == (1, 1)



def test_sample_top_p_reduces_mass():
    logits = torch.randn(1, 50)
    token = sample_top_k_top_p(logits, top_k=0, top_p=0.8, temperature=1.0)
    assert token.shape == (1, 1)
