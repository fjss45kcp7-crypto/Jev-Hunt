import pytest

from jev_hunt.evidence import EvidenceItem, EvidenceRecord
from jev_hunt.policy import CorrectionPolicy, JevDecision


def example_record():
    return EvidenceRecord(
        "synthetic_video", 0, 0.0, 2.0,
        claims={"contact": "Did the objects make contact?"},
        facts=(EvidenceItem("initial-1", "contact", "Contact is occluded.",
                            (0,), 1.0, "unknown", False, "initial"),),
        unresolved_claim_ids=("contact",),
    )


def new_fact(polarity="anomaly"):
    return EvidenceItem(
        "new-1", "contact", "Visible contact at t=1.1",
        (5,), 1.1, polarity, True, "rescan"
    )


def decision(state, probability, conf=0.9):
    return JevDecision(state, "stop", "contact", probability, conf)


def test_positive_and_negative_updates_are_bounded_and_asymmetric():
    record = example_record()
    policy = CorrectionPolicy(kappa=0.35, max_logit_shift=1.0)
    with_anomaly = record.with_rescan((new_fact("anomaly"),))
    pos = policy.correct(
        0.0, record, with_anomaly, decision("supports_anomaly", 0.99),
        requested_focus_id="contact"
    )
    assert pos.changed and pos.corrected_logit == pytest.approx(0.35)
    with_normal = record.with_rescan((new_fact("normal"),))
    neg = policy.correct(
        0.0, record, with_normal, decision("supports_normal", 0.01),
        requested_focus_id="contact"
    )
    assert neg.changed and neg.corrected_logit == pytest.approx(-0.35)


def test_missing_or_conflicting_evidence_preserves_original():
    record = example_record()
    policy = CorrectionPolicy()
    for updated, post in (
        (record, decision("supports_anomaly", 0.99)),
        (record.with_rescan((new_fact("anomaly"),)), decision("conflict", 0.9)),
        (record.with_rescan((new_fact("anomaly"),)), decision("supports_normal", 0.01)),
        (record.with_rescan((new_fact("normal"),)), decision("supports_normal", 0.01, 0.1)),
    ):
        result = policy.correct(0.4, record, updated, post, requested_focus_id="contact")
        assert result.corrected_logit == 0.4
        assert not result.changed


def test_no_ungrounded_or_duplicate_evidence_can_be_appended():
    record = example_record()
    with pytest.raises(ValueError):
        record.with_rescan((
            EvidenceItem("initial-1", "contact", "new", (0,), 1.0,
                         "anomaly", True, "rescan"),
        ))
    with pytest.raises(ValueError):
        record.with_rescan((
            EvidenceItem("new-2", "contact", "outside", (0,), 3.0,
                         "anomaly", True, "rescan"),
        ))


def test_confidence_is_not_ground_truth():
    policy = CorrectionPolicy()
    record = example_record()
    result = policy.correct(
        0.0, record, record.with_rescan((new_fact("anomaly"),)),
        decision("supports_anomaly", 0.99, conf=0.2),
        requested_focus_id="contact",
    )
    assert not result.changed
