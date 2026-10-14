# Experimental plan and falsification protocol

This file is a pre-implementation research plan. **No quantitative improvement has been measured here.**

## Baseline validity comes first

1. Reproduce HeadHunt's raw segment scores, not only its final thresholded labels. Report selected heads, 1% LR calibration samples, model details and source splits.
2. Audit the HeadHunt ablation where changing the final threshold appears to affect AUC/AP. ROC-AUC and AP over a fixed continuous score vector are threshold-independent. Check whether post-processing, thresholded predictions or mismatched score arrays were involved before attempting to replicate published numbers.
3. Audit released frame evaluator \`frame_level.py\`: a chain of \`dict.get(...) or...\` discards genuine 0.0 scores; sigmoid on already-probability scores changes probability semantics (although monotone ranking is normally preserved). Make both original/repaired evaluators available as diagnostics; do not quietly conflate them.
4. Freeze a cleaned baseline score pipeline, thresholds, masks, timestamps and annotation conventions. All methods should use exactly the same evaluation code.

## Research questions (each independently testable)

**RQ1 — frame choice:** At identical frame counts, do signed LR-aligned attention/value contributions identify human-verified decisive frames more reliably than (a) uniformly spaced, (b) random, (c) attention-only, (d) |value|-only, (e) MDP3 or other training-free diverse frame selection?

**RQ2 — Jev action:** For the same initial observation records, does Jev select a rescan action/focus which yields more human-verified evidence-gap resolution than (a) deterministic heuristics, (b) random action, or (c) the frozen InternVL3 asked the same bounded multiple-choice questions?

**RQ3 — net detection:** After full-coverage gated correction, do AUC/AP/event-level false alarms improve *without* recall decline, unacceptable latency, or costs hidden by selective coverage?

## Datasets and splits

- Primary: standard **UCF-Crime** frame-level ROC-AUC and **XD-Violence** frame-level AP, plus both scores on both datasets where meaningful. Honor official video/disjoint-source splits and annotation fps.
- Stress tests: **UBnormal** open-set shifts; Street Scene or human-audited hard negatives (normal running, crowds, handheld camera, benign physical contact).
- Cross-dataset test: select heads/LR and set ALL Jev rules on source validation only; no target labels or target-specific tuning during inference.
- Keep development, Jev prompt/policy tuning, and held-out test separate at the video/source-group level. Even without backprop, tuning thresholds on labels uses supervision.
- Prepare a small blinded annotation subset for *decisive-frame retrieval* and *new fact correctness*; labeler cannot see head heatmap, router choice or predicted anomaly score.

## Budget parity and stress

Pre-register beta in {0, .05, .10, .20, 1.0}, low-score audit fraction initially 0.20, and at most one follow-up rescan per reviewed segment. Match:
- number of video frames viewed, crop size and total visual tokens;
- number of InternVL3 forward calls and text tokens generated;
- Jev request count (or comparable baseline question count);
- end-to-end compute, GPU time, remote p50/p95 response latency, API failure rate and costs.

Separate *offline lookahead* from *causal streaming*. Online methods cannot retrieve future frames beyond the decision time.

## Essential matrix

| ID | Variant | New visual observation? | Jev? | Head contribution? |
|---|---|---:|---:|---:|
| B0 | Original HeadHunt | No | No | No |
| B1 | Fixed descriptions + Jev Noul | No | Yes | No |
| B2 | Simple HeadHunt/Jev score mean | No | Yes | No |
| B3 | Uniform frames + Jev + gate | Yes | Yes | No |
| B4 | Random frames + Jev + gate | Yes | Yes | No |
| B5 | Attention-weight frames + Jev + gate | Yes | Yes | Attention only |
| B6 | Signed head frames + fixed rescan action + gate | Yes | No | Yes |
| B7 | Signed head frames + frozen InternVL3 structured choice + gate | Yes | No | Yes |
| B8 | Signed head frames + Jev without rescan | No | Yes | Yes |
| Ours | Signed head frames + Jev rescan + gate | Yes | Yes | Yes |
| A1 | Ours without low-score audits | Yes | Yes | Yes |
| A2 | Ours without conservative normal gate | Yes | Yes | Yes |
| A3 | Ours with different decision option order | Yes | Yes | Yes |

For B7 use constrained token-likelihood/binary or equivalent typed choice, not a purposely weak free-form numerical generation prompt.

## Score integrity and decision accounting

- Main ROC-AUC and AP: **all video frames**, never filter unresolved/abstained frames.
- Report event-level precision/recall, false alarms/hour, event boundaries, and ordinary negative cases; choose operating threshold only on validation.
- Report transitions: correct -> wrong, wrong -> correct; false-positive corrections vs false-negative corrections, each separated by direction.
- Report reliability diagrams, Brier, NLL for scores explicitly treated as probabilities. Do not call a Noul-derived value a calibrated anomaly posterior without evidence.
- Coverage/risk: supplementary graphs for abstention and manual-review routing, alongside **full coverage** scores.
- Confidence interval: video/source-group clustered bootstrap (avoid frame-wise pseudo-replication), with paired per-video deltas.

## Failure-case audit

Tag each reviewed interval: never-sampled event, small target, object hallucination, wrong action order, cross-frame conflation, correlated caption hallucination, wrong Jev option, network/timeouts, incomplete frame mapping, invalid LR reconstruction, or smoothing-induced boundary shift.

For every update, store baseline logit, corrected logit, pre-focused claim, action, observed frame IDs/time, status, whether new fact was visible, and gate reason. For sensitive data store anonymized IDs; never dump vendor credentials.

## Suggested execution order

1. Offline pure-unit tests and original baseline integrity audit.
2. Implement exact selected-head attention/value hook with reconstruction check, followed by human-verified frame recall@B.
3. Implement frozen InternVL3 rescan adapters and blinded atomic-fact validation.
4. Compare Jev routing with constrained InternVL3 routing under matched budgets.
5. Run full-coverage detection evaluation only after RQ1 and RQ2 checks pass.
6. Check cross-domain, strict online and adverse prompts/option-order controls.

## Publication condition

If RQ1 fails, head-guided evidence proposal is not substantiated. If RQ2 fails, Jev is not demonstrated necessary. If RQ3 fails, do not label the system state of the art based on anecdotal videos. Negative results are still scientifically useful.
