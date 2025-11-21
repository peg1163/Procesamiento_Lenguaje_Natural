
import math
from typing import List, Optional

import torch
import torch.nn.functional as F


def _prepare_prompt_ids(prompt, tokenizer, device):
    
    if isinstance(prompt, str):
        ids = tokenizer.encode(prompt)
    else:
        ids = prompt

    if isinstance(ids, torch.Tensor):
        x = ids.to(device)
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [1, T]
    else:
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    return x  # [1, T]


@torch.no_grad()
def decode_greedy(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    device: Optional[torch.device] = None,
) -> str:

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = _prepare_prompt_ids(prompt, tokenizer, device)

    for _ in range(max_new_tokens):
        logits = model(x)  # [1, T, V]
        next_logits = logits[:, -1, :]  # [1, V]
        next_id = torch.argmax(next_logits, dim=-1)  # [1]
        next_id = next_id.unsqueeze(-1)  # [1, 1]
        x = torch.cat([x, next_id], dim=1)  # [1, T+1]

    # Decodificar TODA la secuencia (prompt + generacion)
    ids = x[0].tolist()
    text = tokenizer.decode(ids)
    return text


def _sample_top_k_top_p(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError("temperature debe ser > 0")

    logits = logits / temperature

    # TOP-K
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        min_topk = v[-1]
        logits = torch.where(logits < min_topk, torch.full_like(logits, -float("inf")), logits)

    # TOP-P
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)

        # enmascarar todo lo que supera top_p
        cutoff = cumulative > top_p
        cutoff[..., 0] = False  # siempre dejamos el más probable
        sorted_logits = torch.where(cutoff, torch.full_like(sorted_logits, -float("inf")), sorted_logits)

        # desordenar de vuelta
        logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)  # [1]
    return next_id


@torch.no_grad()
def decode_topk_topp(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> str:

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = _prepare_prompt_ids(prompt, tokenizer, device)

    for _ in range(max_new_tokens):
        logits = model(x)  # [1, T, V]
        next_logits = logits[0, -1, :]  # [V]
        next_id = _sample_top_k_top_p(
            next_logits,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )  # [1]
        x = torch.cat([x, next_id.view(1, 1)], dim=1)

    ids = x[0].tolist()
    text = tokenizer.decode(ids)
    return text


@torch.no_grad()
def decode_beam_search(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    beam_size: int = 4,
    length_penalty: float = 0.7,
    device: Optional[torch.device] = None,
) -> str:
 
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    base = _prepare_prompt_ids(prompt, tokenizer, device)[0]  # [T]
    beams = [(base, 0.0)]  # (tensor [T], logprob)

    def score(seq: torch.Tensor, logprob: float) -> float:
        length = seq.size(0)
        lp = ((5 + length) / 6) ** length_penalty
        return logprob / lp

    for _ in range(max_new_tokens):
        new_beams = []
        for seq, logp in beams:
            x = seq.unsqueeze(0)  # [1, T]
            logits = model(x)  # [1, T, V]
            next_logits = logits[0, -1, :]  # [V]
            log_probs = F.log_softmax(next_logits, dim=-1)  # [V]

            topk_logp, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

            for lp_tok, tok_id in zip(topk_logp, topk_ids):
                new_seq = torch.cat([seq, tok_id.view(1)], dim=0)  # [T+1]
                new_logp = logp + lp_tok.item()
                new_beams.append((new_seq, new_logp))

        # nos quedamos con los mejores beams
        new_beams.sort(key=lambda s: score(s[0], s[1]), reverse=True)
        beams = new_beams[:beam_size]

    
    best_seq, best_logp = max(beams, key=lambda s: score(s[0], s[1]))
    ids = best_seq.tolist()
    text = tokenizer.decode(ids)
    return text
