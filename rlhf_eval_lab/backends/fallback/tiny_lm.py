# rlhf_eval_lab/backends/fallback/tiny_lm.py
# GRU 固定の TinyLM（transformers 非依存）

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRULM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        x = self.emb(input_ids)          # [B, T, E]
        h, _ = self.gru(x)               # [B, T, H]
        logits = self.lm_head(h)         # [B, T, V]
        return logits

    def _forward_with_hidden(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, h_n) for full sequence.
        logits: [B, T, V]
        h_n: [num_layers, B, H] (final hidden state)
        """
        x = self.emb(input_ids)          # [B, T, E]
        h, h_n = self.gru(x)             # h: [B, T, H], h_n: [L, B, H]
        logits = self.lm_head(h)         # [B, T, V]
        return logits, h_n

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Fast autoregressive generation with hidden-state caching.

        Old version recomputed forward(out) every step (O(T^2)).
        New version:
          - run prompt once to get final hidden state
          - then step one token at a time using cached hidden (O(T + K))
        """
        self.eval()

        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B,T], got shape={tuple(input_ids.shape)}")

        out = input_ids.clone()

        # 1) Encode prompt once
        logits, h = self._forward_with_hidden(out)     # logits: [B,T,V], h: [L,B,H]

        # Handle edge: if prompt length is 0 (shouldn't happen usually)
        if out.size(1) == 0:
            # Create a dummy token 0 to start; still deterministic in shape.
            # (Caller should avoid empty prompts; this is just a guard.)
            bsz = out.size(0)
            out = torch.zeros((bsz, 1), dtype=torch.long, device=out.device)
            logits, h = self._forward_with_hidden(out)

        # 2) Autoregressive steps (cached hidden)
        for _ in range(int(max_new_tokens)):
            last_logits = logits[:, -1, :]  # [B, V]
            denom = max(1e-6, float(temperature))
            last_logits = last_logits / denom

            probs = F.softmax(last_logits, dim=-1)

            # sampling (caller can set torch.manual_seed(seed) for determinism)
            next_id = torch.multinomial(probs, num_samples=1)  # [B,1]

            # append
            out = torch.cat([out, next_id], dim=1)

            # step GRU by 1 token using cached hidden state
            x_next = self.emb(next_id)              # [B,1,E]
            h_step, h = self.gru(x_next, h)         # h_step: [B,1,H], h: [L,B,H]
            logits_next = self.lm_head(h_step)      # [B,1,V]

            # update logits history (cheap append)
            logits = torch.cat([logits, logits_next], dim=1)

        return out

    def logprob_of_sequence(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 合計 log p(x_t | x_<t>)
        # input_ids: [B, T]
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B,T], got shape={tuple(input_ids.shape)}")

        # predict next-token logits for positions 0..T-2
        logits = self.forward(input_ids[:, :-1])     # [B, T-1, V]
        logp = F.log_softmax(logits, dim=-1)         # [B, T-1, V]
        tgt = input_ids[:, 1:]                       # [B, T-1]
        lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        return lp.sum(dim=1)                         # [B]

    def nll_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B,T], got shape={tuple(input_ids.shape)}")

        logits = self.forward(input_ids[:, :-1])  # [B, T-1, V]
        tgt = input_ids[:, 1:]                    # [B, T-1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            reduction="mean",
        )
        return loss
