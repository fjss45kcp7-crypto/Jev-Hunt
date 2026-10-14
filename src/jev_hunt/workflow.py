"""Small offline/online interface around a frozen detector and frozen evidence extractor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence
import math

from .budget import select_review_candidates
from .evidence import EvidenceItem, EvidenceRecord
from .policy import CorrectionPolicy, JevDecision, sigmoid


class DecisionClient(Protocol):
    def decide(
        self, record: EvidenceRecord, *, phase: str,
        focus_id: str | None = None,
    ) -> JevDecision: ...


class EvidenceProvider(Protocol):
    def rescan(
        self, record: EvidenceRecord, *, action: str, focus_id: str,
    ) -> Sequence[EvidenceItem]: ...


@dataclass(frozen=True)
class SegmentAudit:
    index: int
    selection_reason: str
    correction_reason: str
    selected_action: str
    focus_id: str
    original_logit: float
    corrected_logit: float
    new_fact_count: int

    def to_dict(self) -> dict:
        return vars(self).copy()


@dataclass(frozen=True)
class VideoReviewResult:
    original_logits: tuple[float, ...]
    corrected_logits: tuple[float, ...]
    audits: tuple[SegmentAudit, ...]

    def to_dict(self) -> dict:
        return {
            "original_logits": list(self.original_logits),
            "corrected_logits": list(self.corrected_logits),
            "original_scores": [sigmoid(s) for s in self.original_logits],
            "corrected_scores": [sigmoid(s) for s in self.corrected_logits],
            "audits": [a.to_dict() for a in self.audits],
            "total_segments": len(self.original_logits),
            "total_reviewed": len(self.audits),
            "total_corrected": sum(a.original_logit != a.corrected_logit for a in self.audits),
            "evaluation_note": "Score all frames/videos; never drop unresolved segments.",
        }


class ReviewWorkflow:
    """Pluggable reviewer; inference only, no training or adaptation.

    An EvidenceProvider for actual InternVL3 rescans is supplied by the caller.
    Bad API responses, missing context and provider failures FAIL OPEN TO THE
    ORIGINAL LOGIT (not to a predicted normal label).
    """

    def __init__(
        self,
        decider: DecisionClient,
        provider: EvidenceProvider,
        *,
        policy: CorrectionPolicy | None = None,
    ) -> None:
        self.decider = decider
        self.provider = provider
        self.policy = policy or CorrectionPolicy()

    def run(
        self,
        logits: Sequence[float],
        records: Mapping[int, EvidenceRecord],
        *,
        review_fraction: float = 0.10,
        low_score_fraction: float = 0.20,
        threshold: float = 0.65,
        seed: int = 42,
    ) -> VideoReviewResult:
        original = tuple(float(x) for x in logits)
        if any(not math.isfinite(x) for x in original):
            raise ValueError("baseline logits must be finite")
        scores = [sigmoid(x) for x in original]
        candidates = select_review_candidates(
            scores, review_fraction=review_fraction,
            low_score_fraction=low_score_fraction, threshold=threshold, seed=seed
        )
        updated = list(original)
        audits = []
        for item in candidates:
            index = item.index
            original_logit = original[index]
            record = records.get(index)
            reason, action, focus, extra = "missing_initial_record", "stop", "none", 0
            if record is not None:
                if record.segment_index != index:
                    raise ValueError("manifest record index does not match segment index")
                try:
                    pre = self.decider.decide(record, phase="pre")
                    action, focus = pre.action, pre.focus_id
                    if action == "stop":
                        reason = "no_additional_observation"
                    elif focus == "none" or focus not in record.claims:
                        reason = "invalid_focus"
                    else:
                        new_facts = tuple(self.provider.rescan(
                            record, action=action, focus_id=focus
                        ))
                        extra = len(new_facts)
                        augmented = record.with_rescan(new_facts)
                        post = self.decider.decide(
                            augmented, phase="post", focus_id=focus
                        )
                        correction = self.policy.correct(
                            original_logit, record, augmented, post,
                            requested_focus_id=focus
                        )
                        updated[index] = correction.corrected_logit
                        reason = correction.reason
                except Exception as exc:
                    # Do not leak videos, prompts or API keys in exception logs.
                    reason = "review_error_" + type(exc).__name__
                    updated[index] = original_logit
            audits.append(SegmentAudit(
                index, item.reason, reason, action, focus,
                original_logit, updated[index], extra
            ))
        return VideoReviewResult(original, tuple(updated), tuple(audits))
