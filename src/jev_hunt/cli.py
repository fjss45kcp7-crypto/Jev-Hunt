"""Run cached-evidence experiments; does not invent live video observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .evidence import EvidenceItem, EvidenceRecord
from .jev_client import JevClient
from .policy import JevDecision, CorrectionPolicy
from .workflow import ReviewWorkflow


class NoopDecider:
    """Baseline-only dry-run: never masquerades as real Jev."""
    def decide(self, record: EvidenceRecord, *, phase: str,
               focus_id: str | None = None) -> JevDecision:
        return JevDecision(
            "insufficient", "stop", focus_id or "none", 0.5, 0.0
        )


class ManifestProvider:
    """Read pre-recorded, timestamped rescan facts; NO image/video inference."""
    def __init__(self, rescans: dict[str, Any]) -> None:
        self.rescans = rescans

    def rescan(
        self, record: EvidenceRecord, *, action: str, focus_id: str,
    ) -> Sequence[EvidenceItem]:
        entries = self.rescans.get(str(record.segment_index), {}).get(action, [])
        return tuple(
            EvidenceItem(**{**item, "frame_ids": tuple(item["frame_ids"]), "phase": "rescan"})
            for item in entries
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Frozen baseline + budgeted text-evidence review; no model training"
    )
    parser.add_argument("manifest", type=Path, help="JSON cached-evidence manifest")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="No API calls; logits unchanged (default)")
    modes.add_argument("--live-jev", action="store_true", help="Real hosted Jev calls; needs TYPESAFE_API_KEY")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--review-fraction", type=float, default=0.10)
    parser.add_argument("--low-score-fraction", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kappa", type=float, default=0.35)
    parser.add_argument("--max-logit-shift", type=float, default=1.0)
    parser.add_argument("--jev-model", default="jev-latest")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    logits = manifest["logits"]
    records = {
        record.segment_index: record
        for record in (EvidenceRecord.from_dict(x) for x in manifest["records"])
    }
    decider = JevClient(model=args.jev_model) if args.live_jev else NoopDecider()
    try:
        result = ReviewWorkflow(
            decider, ManifestProvider(manifest.get("rescans", {})),
            policy=CorrectionPolicy(kappa=args.kappa,
                                    max_logit_shift=args.max_logit_shift)
        ).run(
            logits, records, review_fraction=args.review_fraction,
            low_score_fraction=args.low_score_fraction,
            threshold=args.threshold, seed=args.seed
        )
    finally:
        if isinstance(decider, JevClient):
            decider.close()
    output = result.to_dict()
    output["mode"] = "live_jev" if args.live_jev else "baseline_only_dry_run"
    output["input_kind"] = "cached_text_evidence_not_raw_video"
    serialized = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
