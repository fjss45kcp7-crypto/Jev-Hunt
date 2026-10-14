# Jev-Hunt: method specification (research proposal)

**Version:** 2026-10-14. **Status:** proposed method + modular prototype, NOT reported VAD results.

## 1. Objective and invariants

Build an *additional inference-time decision layer* around an **existing, frozen** HeadHunt-VAD detector. Do not retrain Jev, InternVL3, HeadHunt's robust-head identification (RHI), or its logistic-regression (LR) scorer. Reuse the published head set and original LR weights. No new trainable router, fusion MLP, or learned temporal module.

This is *no-additional-training*, **not** an assertion that the full original HeadHunt pipeline is completely unsupervised: the original RHI/LR/threshold calibration uses labeled data. Count all selection and calibration labels in reported budgets.

Jev (TypeSafe System One) accepts text/JSON only. It is used for **which evidence is insufficient and what to inspect next**, not for direct video/feature inference. Its Noul output is **not presumed calibrated** on video anomalies.

## 2. Baseline to preserve

For original selected heads I*, each of width d:

    z_t = concat(h[t,k] for k in I*)
    ell_t^H = b + w.T @ z_t
    p_t^H = sigmoid(ell_t^H)

Freeze K, extraction order, weights and bias. Evaluate the original per-segment continuous logits before new postprocessing; keep the original Gaussian smoothing and threshold unchanged in primary comparisons. If source implementation does not expose the exact logits, derive them from its saved LR and audit equivalence with original predict_proba.

The original per-prompt RHI criterion concerns *dataset-level head saliency stability*. It does not establish input-level correctness, causal attribution, invariance to cameras, or robustness to unseen anomalies.

## 3. Head-grounded *signed* evidence proposal

Given the actual readout query token q, for selected head k and each available key token j:

    h[t,k] = sum_j alpha[t,k,q,j] * v[t,k,j]
    r[t,k,j] = alpha[t,k,q,j] * (w_k.T @ v[t,k,j])
    ell_t^H = b + sum_k sum_j r[t,k,j]

For visual tokens V_f mapped to frame f:

    R_plus[t,f]  = sum_{k,j in V_f} max(r[t,k,j], 0)
    R_minus[t,f] = sum_{k,j in V_f} max(-r[t,k,j], 0)

Rank both positive and negative contributions plus distributed temporal context; never erase text-token contributions from the logit reconstruction. Positive and negative are **algebraic contributions to the LR decision**, not proven anomaly/normal causal evidence. Deep token representations mix context; do not call a high-scoring visual token the ground-truth location.

**Strict extraction prerequisites:**
- Capture the correct q at the o_proj **pre-hook** / baseline selected-head readout, not an arbitrary generation token.
- Reconstruct attention softmax after the exact mask and relative/rotary position handling, GQA key/value-head replication and head mapping.
- Use the matching value-projection tensor at the same forward pass and correctly map visual keys to frames (and optional patches).
- Check the original LR logit against b + sum(r) with strict numerical tolerance. If it fails, mark attribution invalid and fall back to uniform/context sampling. Do not silently treat it as genuine attribution.
- Account separately for text, BOS, image delimiters, padding and other tokens; frame ids alone are insufficient without verified processor metadata.
- This extra selective attention computation is **not** a costless single-forward-pass promise.

A correctly reconstructed identity is still not a causal-intervention result. Include attention-only, norm-only, random, uniform, temporal-diverse and signed-contribution ablations.

## 4. Budgeted review scheduler

Run the original head scorer across *all* segments. For n segments, fix B = ceil(beta*n) with beta=0.10 as an initial **hypothesis**, not a tuned optimum. Partition the review set across (a) near-threshold, (b) high-score alarms, and (c) random low-score audits. The latter is required because a policy that only inspects detections cannot catch high-confidence baseline false negatives.

For prototype reproducibility, see budget.py. Pre-register the seed and fractions or tune only on a disjoint validation set. Report alternative budgets beta in {0, .05, .10, .20, 1.0} and absolute extra compute.

## 5. Evidence interface: from frozen visual model to Jev

For each reviewed interval build a small, timestamp-grounded record with:
- a finite set of **possible questions** (e.g. real contact, object-state change, occlusion);
- atomic observations with frame id, source timestamp, visibility, anomaly/normal/unknown polarity and a phase ('initial' or 'rescan');
- explicitly unresolved claims; and
- no ground-truth labels, class-bearing path names, baseline score, or invented unseen visual facts.

The same frozen InternVL3 supplies the observations. Keep prompts neutral to the original alarm to avoid confirmation bias. Distinguish **not observed** from **observed absent**. Never infer that the model's short description is independently validated: generated observations can be correlated, hallucinatory, or inaccurate.

Jev evaluates typed questions on bounded JSON:
- status Choice: supports_anomaly / supports_normal / insufficient / conflict;
- pre-review focus Choice: one known claim id or none;
- pre-review action Choice: temporal_dense / zoom_with_context / temporal_context / stop;
- Noul: does available visible evidence substantiate an abnormal event?

In each request, questions are answered against the same state independently; do not assume chain-of-thought or sequential dependence *within* the request. A second request follows a real new visual observation.

A rescan can add multiple frames, a spatial crop *with its original full frame*, or temporal context. In **strict online** evaluation, forbid frames with timestamps later than the decision deadline.

## 6. Evidence-gated correction

Let q^J be the post-rescan Noul answer and ell^H the frozen baseline logit.

    d = clip(logit(clip(q^J, eps, 1-eps)) - ell^H, -L, L)
    ell* = ell^H + kappa * g_direction * d
    p* = sigmoid(ell*)

No claim is made that q^J is a calibrated VAD posterior. Initial kappa=0.35, L=1.0 are **hypotheses**, not empirical optima. The bound |ell* - ell^H| <= kappa*L is algebraic; it does not guarantee improved AUC or safety.

The gate activates only with a new, visible, timestamped observation about the *same pre-declared focus*, an aligned final Jev evidence status, sufficient Choice confidence and a consistent correction direction. Suppression (normal direction) additionally requires focal uncertainty resolution and no freshly observed opposing anomaly evidence. No extra visual evidence, unsupported choice, API error, timeout, occlusion, conflict, uncertainty, or all quotas spent => **the original logit is retained**.

The "visible" flag means the original visual extractor *claims* that its source shows the fact; it is not independent human verification. In high-stakes settings a real human review is needed.

Retain every segment (including unresolved cases) in the full-coverage AUC/AP. Abstention is separately audited, not hidden from ranking metrics.

## 7. Provisional contributions — what is and is not novel

**To test:** (1) mapping signed, frozen LR-attention contributions to targeted frame candidates; (2) whether Jev selects *new* evidence that improves a frozen detector under equal compute budgets and strictly gated corrections.

**Not original alone:** LLM description-based VAD (LAVAD); training-free event prompting (EventVAD); choice/readout via Jev (Decision Readouts...); question-based dynamic VAD (QVAD); staged evidence escalation (PARSEE-VAD); active exploration/verification (AgenticVAU); normal counterfactual checks (MACD); simple confidence or evidence gating (selective classification / TACO / GEASS); adaptive frame selection (AdaFrame / MDP3).

**Critical null hypotheses:** (a) head-aligned frame choice equals attention-only/random; (b) Jev routing equals deterministic or frozen InternVL3 structured choice; (c) changes increase full-coverage errors after equal-budget normalization. Publish negative findings if supported.

## 8. Immediate known limitations

- Jev never sees pixels: it cannot recover an unobserved event without a rescan.
- Rescans use the same InternVL3: errors can be strongly correlated.
- Thresholds and \`kappa\` have no verified calibration without held-out data.
- Our code implements a *tensor-level attribution calculator*, NOT a validated InternVL3 full-attention hook.
- The head-based proposal may not be informative for genuinely unseen anomaly types; random low-score audits remain necessary.
- Privacy: Jev's remote API receives text evidence. Remove file names, identities and sensitive descriptions unless permitted.
- Offline smoothing can depend on future observations; online claims need a separate causal variant.
- Extra model calls and Jev network latency must be measured end-to-end.

## Primary sources

- HeadHunt-VAD: https://arxiv.org/abs/2512.17601 — code: https://github.com/CebCai/HeadHunt-VAD
- Jev official primitives / text-only constraints: https://docs.typesafe.ai/concepts/system-one
- Choice: https://docs.typesafe.ai/primitives/choice ; Noul: https://docs.typesafe.ai/primitives/noul
- Direct Jev-VAD predecessor: https://arxiv.org/abs/2609.34180
- Closest selection/escalation: https://arxiv.org/abs/2609.33236 ; https://arxiv.org/abs/2604.03040
- Value-norm contribution caveat: https://arxiv.org/abs/2004.10102
