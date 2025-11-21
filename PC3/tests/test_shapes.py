import torch

from src.models.transformer import MicroTransformer


def test_transformer_shapes():
    vocab_size = 32
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 128
    max_seq_len = 32

    model = MicroTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.1,
        max_seq_len=max_seq_len,
        posenc_type="rope",
    )

    B, T = 4, 16
    x = torch.randint(0, vocab_size, (B, T))
    logits = model(x)

    assert logits.shape == (B, T, vocab_size)
