from jev_hunt.evidence import EvidenceItem, EvidenceRecord
from jev_hunt.policy import CorrectionPolicy, JevDecision
from jev_hunt.workflow import ReviewWorkflow


def record(index):
    return EvidenceRecord(
        "synthetic", index, float(index), float(index + 1),
        {"contact": "Did contact occur?"},
        (EvidenceItem(f"i{index}", "contact", "Unclear contact",
                      (0,), index + 0.2),),
        ("contact",),
    )


class Decisions:
    def decide(self, record, *, phase, focus_id=None):
        if phase == "pre":
            return JevDecision("insufficient", "temporal_dense", "contact", 0.5, 0.8)
        return JevDecision("supports_anomaly", "stop", focus_id, 0.99, 0.9)


class Rescan:
    def rescan(self, record, *, action, focus_id):
        return [EvidenceItem(
            f"r{record.segment_index}", focus_id, "Visible contact",
            (1,), record.start_s + 0.3, "anomaly", True, "rescan"
        )]


class BrokenRescan:
    def rescan(self, record, *, action, focus_id):
        raise RuntimeError("test intentionally fails")


def test_new_evidence_required_and_original_logits_retained_elsewhere():
    logits = [-1.0] * 5
    records = {i: record(i) for i in range(5)}
    flow = ReviewWorkflow(Decisions(), Rescan(), policy=CorrectionPolicy())
    result = flow.run(logits, records, review_fraction=0.4, seed=11)
    assert len(result.audits) == 2
    assert sum(new != old for new, old in zip(result.corrected_logits, logits)) == 2
    assert len(result.to_dict()["corrected_scores"]) == len(logits)


def test_error_falls_back_to_original_logits():
    logits = [0.0] * 5
    flow = ReviewWorkflow(Decisions(), BrokenRescan())
    result = flow.run(logits, {i: record(i) for i in range(5)}, review_fraction=1)
    assert result.corrected_logits == tuple(logits)
    assert all("review_error_" in a.correction_reason for a in result.audits)


def test_missing_evidence_kept_in_full_coverage_output():
    result = ReviewWorkflow(Decisions(), Rescan()).run(
        [0.0, 0.0], {}, review_fraction=1
    )
    assert len(result.corrected_logits) == 2
    assert all(a.correction_reason == "missing_initial_record" for a in result.audits)
