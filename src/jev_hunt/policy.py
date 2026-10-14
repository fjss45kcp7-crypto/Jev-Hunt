"""Conservative correction of a FROZEN original logit; no learned fusion head."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .evidence import EvidenceRecord

STATES = frozenset({"supports_anomaly", "supports_normal", "insufficient", "conflict"})
ACTIONS = frozenset({"temporal_dense", "zoom_with_context", "temporal_context", "stop"})


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


@dataclass(frozen=True)
class JevDecision:
    state: str
    action: str
    focus_id: str
    anomaly_support: float
    choice_confidence: float

    def __post_init__(self) -> None:
        if self.state not in STATES or self.action not in ACTIONS or not self.focus_id:
            raise ValueError("unknown Jev choice or empty focus id")
        if not (0 <= self.anomaly_support <= 1 and math.isfinite(self.anomaly_support)):
            raise ValueError("invalid Noul probability")
        if not (0 <= self.choice_confidence <= 1 and math.isfinite(self.choice_confidence)):
            raise ValueError("invalid Choice confidence")


@dataclass(frozen=True)
class CorrectionResult:
    original_logit: float
    corrected_logit: float
    original_probability: float
    corrected_probability: float
    changed: bool
    reason: str


@dataclass(frozen=True)
class CorrectionPolicy:
    kappa: float = 0.35
    max_logit_shift: float = 1.0
    min_choice_confidence: float = 0.35
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if not (0 <= self.kappa <= 1 and 0 <= self.max_logit_shift < float("inf")):
            raise ValueError("invalid correction magnitude")
        if not (0 <= self.min_choice_confidence <= 1 and 0 < self.eps < 0.5):
            raise ValueError("invalid confidence or epsilon")

    def correct(
        self,
        original_logit: float,
        original_record: EvidenceRecord,
        updated_record: EvidenceRecord,
        post: JevDecision,
        *,
        requested_focus_id: str,
    ) -> CorrectionResult:
        if not math.isfinite(original_logit):
            raise ValueError("original logit must be finite")
        original_p = sigmoid(original_logit)

        def preserve(reason: str) -> CorrectionResult:
            return CorrectionResult(
                original_logit, original_logit, original_p, original_p, False, reason
            )

        if post.state in {"insufficient", "conflict"}:
            return preserve("unresolved_or_conflicting")
        if post.choice_confidence < self.min_choice_confidence:
            return preserve("low_choice_confidence")
        if (requested_focus_id == "none" or requested_focus_id not in original_record.claims
                or post.focus_id != requested_focus_id):
            return preserve("focus_mismatch")
        old_ids = {f.evidence_id for f in original_record.facts}
        new_facts = [f for f in updated_record.facts
                     if f.evidence_id not in old_ids and f.phase == "rescan"]
        if not new_facts:
            return preserve("no_new_visual_evidence")
        if self.kappa == 0 or self.max_logit_shift == 0:
            return preserve("correction_disabled")
        q = min(1 - self.eps, max(self.eps, post.anomaly_support))
        proposed = math.log(q / (1 - q)) - original_logit
        if post.state == "supports_anomaly":
            if proposed <= 0:
                return preserve("direction_disagreement")
            allowed = any(f.claim_id == requested_focus_id and f.visible
                          and f.polarity == "anomaly" for f in new_facts)
            if not allowed:
                return preserve("missing_new_anomaly_evidence")
        elif post.state == "supports_normal":
            if proposed >= 0:
                return preserve("direction_disagreement")
            allowed = any(f.claim_id == requested_focus_id and f.visible
                          and f.polarity == "normal" for f in new_facts)
            # Normal evidence must rebut the focal question; unknowns and fresh
            # contrary anomaly observations prevent down-correction.
            if not allowed or requested_focus_id in updated_record.unresolved_claim_ids:
                return preserve("normal_evidence_not_resolved")
            if any(f.visible and f.polarity == "anomaly" for f in new_facts):
                return preserve("contradictory_new_anomaly_evidence")
        else:
            return preserve("unsupported_state")
        shift = self.kappa * max(-self.max_logit_shift,
                                 min(self.max_logit_shift, proposed))
        updated_logit = original_logit + shift
        return CorrectionResult(
            original_logit, updated_logit, original_p, sigmoid(updated_logit),
            not math.isclose(updated_logit, original_logit, abs_tol=1e-12),
            "gated_positive" if shift > 0 else "gated_negative"
        )
