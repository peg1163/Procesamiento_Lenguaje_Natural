from pathlib import Path
from typing import Sequence, Tuple, List

VOCAB = "0123456789+()=abc\n"

class CharTokenizer:
    #Tokenizador carácter-a-índice determinista para corpus sintético 
    def __init__(self, vocab: str = VOCAB):
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[i] for i in ids)

def load_corpus(out_dir: str = "out") -> list[str]:
    #Lee el corpus generado por tools/gen_corpus.sh
    raw = Path(out_dir) / "corpus_raw.txt"
    return raw.read_text(encoding="utf-8").splitlines()

def make_splits(lines: Sequence[str],
                split: Tuple[float, float, float] = (0.9, 0.05, 0.05)) -> tuple[list[str], list[str], list[str]]:
    n = len(lines)
    a = int(n * split[0])
    b = int(n * (split[0] + split[1]))
    return list(lines[:a]), list(lines[a:b]), list(lines[b:])
