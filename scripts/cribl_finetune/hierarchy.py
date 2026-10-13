"""Exact techniques under the family head: P(technique) = P(family) x P(technique | family), the
second factor a softmax within each family's own techniques (the parent and its sub-techniques).
Torch only, so it is testable without the model (prereg ``cribl_technique_head_2026-10-14.json``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch


def technique_index(
    families: Sequence[str], texts: Mapping[str, Sequence[Mapping[str, str]]]
) -> tuple[list[str], list[int]]:
    """(every listed family's techniques in catalogue order, the family index of each)."""
    techniques: list[str] = []
    group: list[int] = []
    for position, family in enumerate(families):
        for item in texts[family]:
            techniques.append(str(item["id"]))
            group.append(position)
    return techniques, group


def technique_targets(gold: Iterable[str], techniques: Sequence[str]) -> list[int]:
    """Indices in ``techniques`` of the gold ids that are listed (a parent id is its own option)."""
    index = {technique: position for position, technique in enumerate(techniques)}
    return [index[item] for item in dict.fromkeys(str(g) for g in gold) if item in index]


def within_family_log_probs(logits: torch.Tensor, group: torch.Tensor, count: int) -> torch.Tensor:
    """log P(technique | its family) for (batch, techniques) logits: a log-softmax within each of
    the ``count`` family groups given by ``group`` (the family index of every technique)."""
    index = group.unsqueeze(0).expand_as(logits)
    floor = torch.full(
        (logits.shape[0], count), float("-inf"), device=logits.device, dtype=logits.dtype
    )
    peak = floor.scatter_reduce(1, index, logits.detach(), reduce="amax").gather(1, index)
    total = torch.zeros_like(floor).scatter_add(1, index, (logits - peak).exp())
    return logits - peak - total.gather(1, index).log()


def joint_log_probs(
    family_log_p: torch.Tensor, within: torch.Tensor, group: torch.Tensor
) -> torch.Tensor:
    """log P(family) + log P(technique | family): a distribution over every technique."""
    return family_log_p[:, group] + within


def pick(
    joint: torch.Tensor, family_log_p: torch.Tensor, group: torch.Tensor, considered: int | None
) -> int:
    """The most probable technique (one row) among the ``considered`` most probable families
    (``None``: every family)."""
    if considered is None:
        return int(torch.argmax(joint))
    allowed = torch.zeros_like(family_log_p, dtype=torch.bool)
    allowed[torch.topk(family_log_p, min(considered, family_log_p.numel())).indices] = True
    return int(torch.argmax(joint.masked_fill(~allowed[group], float("-inf"))))
