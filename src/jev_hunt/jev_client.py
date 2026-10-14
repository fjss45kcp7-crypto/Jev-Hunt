"""Optional official TypeSafe SDK interface; Jev is NOT trained by this package.

Jev only reads bounded TEXT/JSON facts. Answers are independent against
the SAME state per request. The updated rescan evidence is sent in a second call.
"""

from __future__ import annotations

from typing import Any, Mapping

from .evidence import EvidenceRecord
from .policy import ACTIONS, STATES, JevDecision

STATUS_CRITERIA = {
    "supports_anomaly": "Time-grounded, visibly observed evidence supports abnormal behavior.",
    "supports_normal": "Time-grounded visible facts support an ordinary explanation for the suspected event.",
    "insufficient": "The decisive action or object is occluded, missing, uncertain or not observed.",
    "conflict": "Visible claims support incompatible conclusions that cannot yet be resolved.",
}
ACTION_CRITERIA = {
    "temporal_dense": "Sample additional timestamps inside the current interval to resolve a brief event.",
    "zoom_with_context": "Inspect a small visible region at higher resolution, retaining the full frame.",
    "temporal_context": "Inspect surrounding timestamps to disambiguate before/after relationships.",
    "stop": "No available observation could reliably resolve the evidence gap or review should abstain.",
}


def parse_jev_answers(
    answers: Mapping[str, Any], *, phase: str, requested_focus_id: str | None = None,
) -> JevDecision:
    """Pure parser; safe to test with recorded SDK responses without an API key."""
    if phase not in {"pre", "post"}:
        raise ValueError("phase must be pre or post")
    status = str(answers["status"].choice)
    confidence = float(answers["status"].confidence)
    support = float(answers["anomaly_support"].noul)
    if phase == "pre":
        action = str(answers["action"].choice)
        focus_id = str(answers["focus"].choice)
    else:
        action = "stop"
        focus_id = requested_focus_id or "none"
    return JevDecision(status, action, focus_id, support, confidence)


class JevClient:
    """Two-stage typed decisions using the published typesafe-sdk.

    No key is needed until an actual live request. SDK obtains TYPESAFE_API_KEY
    from environment. The model alias can be pinned for reproducibility.
    """

    def __init__(self, *, model: str = "jev-latest", client: Any | None = None):
        self.model = model
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from typesafe_sdk import TypeSafeClient
            except ImportError as exc:
                raise ImportError("Install Jev adapter: pip install -e '.[jev]'") from exc
            self._client = TypeSafeClient()
        return self._client

    def decide(
        self, record: EvidenceRecord, *, phase: str,
        focus_id: str | None = None,
    ) -> JevDecision:
        if phase not in {"pre", "post"}:
            raise ValueError("phase must be pre or post")
        if phase == "post" and focus_id not in record.claims:
            raise ValueError("post decision must fix a known original focus id")
        try:
            from typesafe_sdk import Choice, Noul
        except ImportError as exc:
            raise ImportError("Install Jev adapter: pip install -e '.[jev]'") from exc

        # Do not send raw video identifiers, file paths or hidden numeric features.
        state = record.to_state()
        state.pop("video_id", None)
        if len(record.facts) > 48:
            raise ValueError("excessive evidence record; summarize deterministically first")
        state["observations"] = [
            {**fact, "text": fact["text"][:512]}
            for fact in state["observations"]
        ]
        if phase == "post":
            state["question_under_review"] = {
                "claim_id": focus_id,
                "description": record.claims[focus_id],
            }

        questions: dict[str, Any] = {
            "status": Choice(
                instructions=(
                    "Classify the support state of the OBSERVED, timestamped facts only. "
                    "Descriptions are fallible; missing or occluded evidence is insufficient."
                ),
                criteria=STATUS_CRITERIA,
            ),
            "anomaly_support": Noul(
                instructions=(
                    "Do the visible, time-grounded facts substantiate an abnormal "
                    "event in this interval? Unobserved actions are not affirmative evidence."
                ),
            ),
        }
        if phase == "pre":
            questions["action"] = Choice(
                instructions=(
                    "Select ONE feasible next visual observation that best resolves "
                    "the main unknown fact. No unlimited deliberation."
                ),
                criteria=ACTION_CRITERIA,
            )
            questions["focus"] = Choice(
                instructions="Which unresolved claim most needs direct visual verification?",
                criteria={**dict(record.claims),
                          "none": "No currently listed claim can be checked meaningfully."},
            )
        response = self._get_client().system_one(
            model=self.model, state=state, questions=questions
        )
        verdict = parse_jev_answers(response.answers, phase=phase,
                                    requested_focus_id=focus_id)
        if verdict.state not in STATES or verdict.action not in ACTIONS:
            raise ValueError("Jev returned unexpected option")
        if verdict.focus_id != "none" and verdict.focus_id not in record.claims:
            raise ValueError("Jev returned focus outside bounded candidates")
        return verdict

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            if hasattr(self._client, "close"):
                self._client.close()
            self._client = None
