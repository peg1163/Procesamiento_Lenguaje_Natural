from typing import Literal

import torch
import torch.nn as nn

from .attention import make_causal_mask
from .mhsa import MultiHeadSelfAttention
from .posenc import SinusoidalPositionalEncoding, build_alibi_bias

PosencType = Literal["sinusoidal", "rope", "alibi"]

class DecoderBlock(nn.Module):


    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.mhsa = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_rope=use_rope,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        positions: torch.Tensor,
        alibi_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention con pre-norm
        h = self.ln1(x)
        attn_out = self.mhsa(
            h,
            attn_mask=attn_mask,
            positions=positions,
            alibi_bias=alibi_bias,
        )
        x = x + attn_out

        # Feed-forward con pre-norm
        h2 = self.ln2(x)
        ff_out = self.ffn(h2)
        x = x + ff_out
        return x


class MicroTransformer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        posenc_type: PosencType = "rope",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.posenc_type: PosencType = posenc_type

        # Embedding de tokens
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Positional encoding / bias
        self.use_rope = posenc_type == "rope"
        if posenc_type == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model, max_len=max_seq_len
            )
        else:
            self.pos_encoder = None

        if posenc_type == "alibi":
            bias = build_alibi_bias(n_heads, max_seq_len)
            self.register_buffer("alibi_bias", bias, persistent=False)
        else:
            self.register_buffer("alibi_bias", None, persistent=False)

        # Bloques decoder
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_rope=self.use_rope,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
 
        B, T = x.shape
        if T > self.max_seq_len:
            raise ValueError(f"Longitud {T} > max_seq_len {self.max_seq_len}")

        device = x.device
        positions = torch.arange(T, device=device)  # [T]

        # Embeddings
        h = self.token_emb(x)  # [B,T,D]
        if self.pos_encoder is not None:
            h = self.pos_encoder(h)
        h = self.emb_dropout(h)

        # Máscara causal común para todos los bloques
        attn_mask = make_causal_mask(T, device=device)  # [1,1,T,T]

        # Bias ALiBi (si aplica)
        alibi_bias = None
        if self.posenc_type == "alibi" and self.alibi_bias is not None:
            alibi_bias = self.alibi_bias[:, :T, :T]  # [H,T,T]

        # Pila de bloques decoder
        for block in self.blocks:
            h = block(
                h,
                attn_mask=attn_mask,
                positions=positions,
                alibi_bias=alibi_bias,
            )

        h = self.ln_f(h)
        logits = self.lm_head(h)  # [B,T,V]
        return logits
