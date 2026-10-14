import math

import numpy as np
import pytest

from jev_hunt.budget import select_review_candidates


def test_low_score_audit_and_uncertain_high_cases_coexist():
    scores = [0.1] * 60 + [0.64] * 20 + [0.95] * 20
    chosen = select_review_candidates(
        scores, review_fraction=0.1, low_score_fraction=0.2, seed=4
    )
    assert len(chosen) == 10
    assert len({c.index for c in chosen}) == 10
    assert sum(c.reason == "low_score_audit" for c in chosen) == 2
    assert any(c.reason == "near_threshold" for c in chosen)
    assert any(c.reason == "high_score" for c in chosen)
    assert chosen == select_review_candidates(
        scores, review_fraction=0.1, low_score_fraction=0.2, seed=4
    )


def test_small_and_empty_inputs_and_bounds():
    assert select_review_candidates([], review_fraction=0.1) == []
    assert select_review_candidates([0.1, 0.9], review_fraction=0) == []
    assert len(select_review_candidates([0.1, 0.9], review_fraction=1)) == 2
    with pytest.raises(ValueError):
        select_review_candidates([float("nan")])
    with pytest.raises(ValueError):
        select_review_candidates([1.2])
    with pytest.raises(ValueError):
        select_review_candidates([0.5], review_fraction=1.1)
