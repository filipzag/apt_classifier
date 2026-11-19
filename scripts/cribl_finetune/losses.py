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


def yes_no(logits: torch.Tensor, gold: Sequence[int], negative_weight: float) -> torch.Tensor:
    """Each family's own yes/no for one row of family logits (P(family) = sigmoid(logit), not a
    share of one answer): ``-log P`` for every gold family plus ``negative_weight * -log(1 - P)``
    for every other one. Most sentences carry one label although many describe two or three
    techniques, so a missing label is weak evidence of "no" (prereg ``cribl_yesno_2026-10-14.json``)."""
    index = torch.as_tensor(list(gold), device=logits.device, dtype=torch.long)
    others = torch.ones_like(logits, dtype=torch.bool).index_fill(0, index, False)
    positive = torch.nn.functional.softplus(-logits[index]).sum()
    return positive + negative_weight * torch.nn.functional.softplus(logits[others]).sum()
