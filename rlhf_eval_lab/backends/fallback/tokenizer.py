# rlhf_eval_lab/backends/fallback/tokenizer.py
# SimpleTokenizer（空白ベース＋文字フォールバック）
# 重要：fallback tier では「必ず vocab 範囲内」を設計で保証する。
# - 学習/生成は sanity 目的。decode の完全可逆性は不要。

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import hashlib


@dataclass(frozen=True)
class SpecialIds:
    pad: int = 0
    bos: int = 1
    eos: int = 2
    unk: int = 3


class SimpleTokenizer:
    """Deterministic, bounded tokenizer.

    Maps tokens to ids via stable hash modulo (vocab_size - 4).
    This avoids dynamic vocab growth that would break embedding shape.
    """

    def __init__(self, vocab_size: int = 4096, specials: SpecialIds = SpecialIds()):
        if vocab_size <= 8:
            raise ValueError("vocab_size must be > 8")
        self.vocab_size = int(vocab_size)
        self.specials = specials

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        toks = text.strip().split()
        if toks:
            return toks
        return list(text)

    def _hash_to_id(self, tok: str) -> int:
        if not tok:
            return self.specials.unk
        h = hashlib.md5(tok.encode("utf-8")).hexdigest()
        x = int(h, 16)
        return 4 + (x % (self.vocab_size - 4))

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        toks = self.tokenize(text)
        ids = [self._hash_to_id(t) for t in toks]
        if add_special:
            return [self.specials.bos] + ids + [self.specials.eos]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        toks: List[str] = []
        for i in ids:
            ii = int(i)
            if skip_special and ii in (
                self.specials.pad,
                self.specials.bos,
                self.specials.eos,
            ):
                continue
            toks.append(f"t{ii}")
        return " ".join(toks)


def overlap_ratio(prompt: str, completion: str) -> float:
    pt = set((prompt or "").split())
    ct = (completion or "").split()
    if not ct:
        return 0.0
    inter = sum(1 for t in ct if t in pt)
    return float(inter) / float(max(1, len(ct)))
