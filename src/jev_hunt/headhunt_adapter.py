"""Minimal, untrained bridge to HeadHunt-VAD's *existing* readout artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def expert_vector(
    oproj_inputs: np.ndarray,
    head_indices: Sequence[tuple[int, int]],
    *,
    num_heads: int,
    head_dim: int,
) -> np.ndarray:
    """Replicate the original head concatenation order from o_proj pre-hook data.

    Input is [layers, num_heads*head_dim] or [layers, num_heads, head_dim].
    Does NOT select new heads or train a new scoring model.
    """
    x = np.asarray(oproj_inputs)
    if x.ndim == 2 and x.shape[1] == num_heads * head_dim:
        x = x.reshape(x.shape[0], num_heads, head_dim)
    elif x.ndim != 3 or x.shape[1:] != (num_heads, head_dim):
        raise ValueError("invalid HeadHunt o_proj input dimensions")
    if not head_indices:
        raise ValueError("head_indices cannot be empty")
    out = []
    for layer, head in head_indices:
        if layer < 0 or head < 0 or layer >= x.shape[0] or head >= num_heads:
            raise IndexError("head index is outside extractor dimensions")
        out.append(x[layer, head])
    return np.concatenate(out).astype(np.float64, copy=False)


@dataclass(frozen=True)
class FrozenHeadHuntReadout:
    head_indices: tuple[tuple[int, int], ...]
    head_dim: int
    weights: np.ndarray
    bias: float

    @classmethod
    def from_joblib(cls, path: str | Path) -> "FrozenHeadHuntReadout":
        """Read original joblib; never fit or modify the existing LR."""
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Install the optional 'headhunt' dependencies") from exc
        data = joblib.load(path)  # Only load trusted local artifacts; pickle is executable.
        model = data["model"]
        if list(model.classes_) != [0, 1] or model.coef_.shape[0] != 1:
            raise ValueError("expected original binary logistic regression with classes [0, 1]")
        indices = tuple(tuple(pair) for pair in data["head_indices"])
        weights = np.asarray(model.coef_[0], dtype=np.float64)
        head_dim = int(data["head_dim"])
        if weights.shape != (len(indices) * head_dim,):
            raise ValueError("readout weights do not match saved head indices")
        return cls(indices, head_dim, weights, float(model.intercept_[0]))

    def logit(self, flattened_selected_heads: np.ndarray) -> float:
        x = np.asarray(flattened_selected_heads, dtype=np.float64)
        if x.shape != self.weights.shape or not np.isfinite(x).all():
            raise ValueError("selected head vector shape/values mismatch")
        return float(np.dot(self.weights, x) + self.bias)
