"""Frozen, reproducible review scheduling. No learned Jev router."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ReviewCandidate:
    index: int
    score: float
    reason: str


def select_review_candidates(
    scores: Sequence[float],
    *,
    review_fraction: float = 0.10,
    low_score_fraction: float = 0.20,
    threshold: float = 0.65,
    seed: int = 42,
) -> list[ReviewCandidate]:
    """Mix uncertainty, high-risk alerts and random low-score audits.

    Score-only routing cannot guarantee recall of undetected anomalies.
    Keep a low-score audit quota and measure how much it actually recovers.
    """
    p = np.asarray(scores, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("scores must be a finite 1-D array in [0,1]")
    if not (0 <= review_fraction <= 1 and 0 <= low_score_fraction <= 1):
        raise ValueError("budget fractions must be within [0,1]")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be within (0,1)")
    n = len(p)
    count = min(n, math.ceil(n * review_fraction))
    if count == 0:
        return []

    rng = np.random.default_rng(seed)
    selected: dict[int, str] = {}
    low_pool = [i for i in range(n) if p[i] < threshold * 0.5]
    low_count = min(len(low_pool), max(1, round(count * low_score_fraction))
                    if count >= 3 and low_score_fraction > 0 else 0)
    if low_count:
        for i in rng.choice(low_pool, size=low_count, replace=False).tolist():
            selected[int(i)] = "low_score_audit"

    # Near-threshold segments include both sides of the operating decision.
    remaining = count - len(selected)
    uncertain_quota = (remaining + 1) // 2
    by_margin = sorted(range(n), key=lambda i: (abs(float(p[i]) - threshold), i))
    for i in by_margin:
        if len([r for r in selected.values() if r == "near_threshold"]) >= uncertain_quota:
            break
        if i not in selected:
            selected[i] = "near_threshold"

    for i in sorted(range(n), key=lambda i: (-float(p[i]), i)):
        if len(selected) >= count:
            break
        if i not in selected:
            selected[i] = "high_score"

    for i in range(n):
        if len(selected) >= count:
            break
        selected.setdefault(i, "budget_fill")
    return [ReviewCandidate(i, float(p[i]), selected[i]) for i in sorted(selected)]
