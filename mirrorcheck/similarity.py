from __future__ import annotations

from typing import Iterable
import torch
import torch.nn.functional as F


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return cosine similarity between two batches or two feature vectors."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)
    return F.cosine_similarity(x.float(), y.float(), dim=-1)


def mean_similarity(scores: Iterable[float]) -> float:
    values = list(scores)
    if not values:
        raise ValueError("Cannot compute mean similarity over an empty list.")
    return float(sum(values) / len(values))
