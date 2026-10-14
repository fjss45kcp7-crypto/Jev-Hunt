from types import SimpleNamespace

import pytest

from jev_hunt.jev_client import parse_jev_answers
from jev_hunt.policy import JevDecision


def answer(choice=None, confidence=None, noul=None):
    return SimpleNamespace(choice=choice, confidence=confidence, noul=noul)


def test_parse_pre_and_post_typed_answers():
    pre = parse_jev_answers(
        {
            "status": answer(choice="insufficient", confidence=0.6),
            "anomaly_support": answer(noul=0.55),
            "action": answer(choice="temporal_dense"),
            "focus": answer(choice="contact"),
        },
        phase="pre",
    )
    assert pre == JevDecision("insufficient", "temporal_dense", "contact", 0.55, 0.6)
    post = parse_jev_answers(
        {
            "status": answer(choice="supports_anomaly", confidence=0.8),
            "anomaly_support": answer(noul=0.95),
        },
        phase="post", requested_focus_id="contact"
    )
    assert post.focus_id == "contact" and post.action == "stop"


def test_bad_noul_is_rejected_not_silently_clipped():
    with pytest.raises(ValueError):
        parse_jev_answers(
            {
                "status": answer(choice="supports_anomaly", confidence=0.9),
                "anomaly_support": answer(noul=2.0),
            }, phase="post", requested_focus_id="contact"
        )
