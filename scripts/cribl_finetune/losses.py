"""Training-loss pieces of the family head that need only torch (testable without the model)."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def rival_margin(log_p: torch.Tensor, gold: Sequence[int], margin: float) -> torch.Tensor:
    """``max(0, margin - (log P(gold families) - log P(strongest wrong family)))`` for one row of
    family log-probabilities: zero once the gold families beat their closest rival by ``margin``
    nats (prereg ``cribl_family_margin_2026-10-13.json``)."""
    index = torch.as_tensor(list(gold), device=log_p.device, dtype=torch.long)
    gold_log_p = torch.logsumexp(log_p[index], dim=0)
    rivals = log_p.index_fill(0, index, float("-inf"))
    return torch.clamp(margin - (gold_log_p - rivals.max()), min=0.0)
