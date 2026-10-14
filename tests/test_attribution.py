import numpy as np
import pytest

from jev_hunt.attribution import compute_signed_contributions, select_evidence_frames
from jev_hunt.headhunt_adapter import expert_vector


def toy():
    attention = np.array([[0.2, 0.3, 0.5], [0.5, 0.4, 0.1]])
    values = np.array([[[1], [2], [3]], [[-2], [1], [1]]], dtype=float)
    weights = np.array([[2], [-1]], dtype=float)
    # Per-head positions: [.4, 1.2, 3.0] and [1.0, -.4, -.1]
    return attention, values, weights


def test_signed_contributions_reconstruct_original_logit():
    a, v, w = toy()
    result = compute_signed_contributions(
        a, v, w, 0.1, [0, 1, None], expected_logit=5.2
    )
    assert result.reconstructed_logit == pytest.approx(5.2)
    assert result.frame_positive[0] == pytest.approx(1.4)
    assert result.frame_negative[1] == pytest.approx(0.4)
    assert result.text_contribution == pytest.approx(2.9)
    assert select_evidence_frames(result, budget=2) == [0, 1]


def test_bad_alignment_must_fail_closed():
    a, v, w = toy()
    with pytest.raises(ValueError, match="reconstruction"):
        compute_signed_contributions(a, v, w, 0.1, [0, 1, -1], expected_logit=2)
    with pytest.raises(ValueError, match="normalized"):
        compute_signed_contributions(a * 0.7, v, w, 0.1, [0, 1, -1])


def test_negative_and_text_tokens_do_not_invent_frames():
    a, v, w = toy()
    result = compute_signed_contributions(a, v, w, 0.0, [-1, None, -1])
    assert select_evidence_frames(result) == []
    assert result.text_contribution == pytest.approx(5.1)


def test_original_head_order_and_shape_are_preserved():
    x = np.arange(2 * 3 * 2).reshape(2, 6)
    selected = expert_vector(x, [(1, 2), (0, 0)], num_heads=3, head_dim=2)
    assert selected.tolist() == [10, 11, 0, 1]
    with pytest.raises(IndexError):
        expert_vector(x, [(2, 1)], num_heads=3, head_dim=2)
