"""Signed linear-readout contributions; not a causal or pixel-level explanation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AttributionResult:
    token_contributions: np.ndarray  # [K, T], signed
    frame_positive: Mapping[int, float]
    frame_negative: Mapping[int, float]
    text_contribution: float
    reconstructed_logit: float
    residual: float | None


def compute_signed_contributions(
    attention: np.ndarray,
    values: np.ndarray,
    lr_weights: np.ndarray,
    bias: float,
    token_frame_ids: Sequence[int | None],
    *,
    expected_logit: float | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> AttributionResult:
    """
    Calculate r[k,j] = attention[k,q,j] * dot(w[k], value[k,j]).

    attention: softmax row for the SAME readout query, [selected_heads, key_tokens].
    values: post value-projection vectors for those heads/keys, [K,T,D].
    lr_weights: the ORIGINAL trained LR weights, reshaped [K,D].
    token_frame_ids: [T]; None or -1 identifies non-video tokens.

    Correctness requires the selected heads to share the query token used for
    HeadHunt scoring, matching rotary positions, mask, GQA mapping, and tokenization.
    The optional expected_logit checks the full score reconstruction. Do not
    interpret mapped scores as causal explanations.
    """
    a = np.asarray(attention, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(lr_weights, dtype=np.float64)
    if a.ndim != 2 or v.ndim != 3 or w.ndim != 2:
        raise ValueError("expected attention [K,T], values [K,T,D], weights [K,D]")
    k, t = a.shape
    if v.shape[:2] != (k, t) or w.shape != (k, v.shape[2]):
        raise ValueError("incompatible head/token dimensions")
    if len(token_frame_ids) != t:
        raise ValueError("one frame mapping entry is required per key token")
    if not np.isfinite(a).all() or not np.isfinite(v).all() or not np.isfinite(w).all():
        raise ValueError("nonfinite tensor values")
    if not (np.isfinite(bias) and (a >= 0).all()):
        raise ValueError("invalid attention or bias")
    if not np.allclose(a.sum(axis=1), 1.0, atol=atol, rtol=rtol):
        raise ValueError("attention must be normalized per head")

    contributions = a * np.einsum("ktd,kd->kt", v, w)
    pos: dict[int, float] = {}
    neg: dict[int, float] = {}
    text_score = 0.0
    for j, fid in enumerate(token_frame_ids):
        head_scores = contributions[:, j]
        if fid is None or fid == -1:
            text_score += float(head_scores.sum())
            continue
        if not isinstance(fid, (int, np.integer)) or fid < 0:
            raise ValueError("frame mappings must be nonnegative integers, -1, or None")
        fid = int(fid)
        pos[fid] = pos.get(fid, 0.0) + float(np.maximum(head_scores, 0).sum())
        neg[fid] = neg.get(fid, 0.0) + float(np.maximum(-head_scores, 0).sum())

    reconstructed = float(bias + contributions.sum())
    residual = None
    if expected_logit is not None:
        if not np.isfinite(expected_logit):
            raise ValueError("expected logit must be finite")
        residual = float(reconstructed - expected_logit)
        if not np.isclose(reconstructed, expected_logit, atol=atol, rtol=rtol):
            raise ValueError(
                f"readout reconstruction failed: residual={residual:.6g}; "
                "check hook location / key mapping / model math"
            )
    return AttributionResult(contributions, pos, neg, text_score, reconstructed, residual)


def select_evidence_frames(result: AttributionResult, budget: int = 6) -> list[int]:
    """Select positive, negative and temporally spread context frames without training."""
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    frame_ids = sorted(set(result.frame_positive) | set(result.frame_negative))
    if budget == 0 or not frame_ids:
        return []
    chosen: list[int] = []
    pos_rank = sorted(frame_ids, key=lambda f: (-result.frame_positive.get(f, 0), f))
    neg_rank = sorted(frame_ids, key=lambda f: (-result.frame_negative.get(f, 0), f))

    def add(fid: int) -> None:
        if len(chosen) < budget and fid not in chosen:
            chosen.append(fid)

    if result.frame_positive.get(pos_rank[0], 0) > 0:
        add(pos_rank[0])
    if result.frame_negative.get(neg_rank[0], 0) > 0:
        add(neg_rank[0])

    # Include temporal context even if contribution is concentrated in one frame.
    if budget > len(chosen):
        n_context = min(len(frame_ids), max(1, budget // 3))
        for idx in np.linspace(0, len(frame_ids) - 1, n_context, dtype=int):
            add(frame_ids[int(idx)])

    for fid in pos_rank + neg_rank + frame_ids:
        add(fid)
    return sorted(chosen)
