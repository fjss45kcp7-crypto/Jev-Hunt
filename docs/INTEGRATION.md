# Frozen HeadHunt / InternVL3 / Jev integration guide

**Scope:** The included CLI works on **cached text evidence**. The original raw-video feature extraction and a new frozen-model visual rescan still need a model-specific adapter. Do not present the synthetic manifest as a working video benchmark.

## 1. Obtain baseline artifacts from the original project

Install and run the independent official HeadHunt-VAD implementation (the project is not vendored or forked here):

- https://github.com/CebCai/HeadHunt-VAD
- Paper: https://arxiv.org/abs/2512.17601

Follow its own extraction, RHI and LR fitting instructions. Save the selected expert heads and **existing** scorer joblib. Never fit an extra Jev, classifier, VLM, gate or feature selector. Preserve model weights, prompts, processing dimensions and ordering as exact experiment metadata.

The released InternVL extractor hooks the **input** to each attention \`o_proj\` module and takes the last sequence position captured. Inspect whether generation in your model build changes this query location; save the exact query index and input-token metadata.

## 2. Reproduce the original LR logit

\`\`\`python
from jev_hunt.headhunt_adapter import FrozenHeadHuntReadout, expert_vector

readout = FrozenHeadHuntReadout.from_joblib("models/anomaly_scorer.joblib")
# oproj_inputs must be output from the ORIGINAL HeadHunt extractor:
z = expert_vector(
    oproj_inputs, readout.head_indices, num_heads=NUM_HEADS,
    head_dim=readout.head_dim
)
logit = readout.logit(z)
\`\`\`

For binary sklearn LR with classes [0,1], compare \`sigmoid(logit)\` with original \`predict_proba(z[None,:])[:,1]\`. **Do not** apply the original sigmoid twice or silently change smoothing. \`joblib.load\` uses pickle internally: only load trusted artifacts.

## 3. Model-specific contribution extraction is required

The pure NumPy primitive accepts:
- \`attention[K,T]\` = post-softmax attention at **the exact original query q**;
- \`values[K,T,D]\` = selected per-head value vectors with correct GQA/head mapping;
- \`lr_weights[K,D]\` = frozen original readout weights;
- \`token_frame_ids[T]\` from *processor tokenization*, with \`None\` for non-video positions;
- \`expected_logit\` computed from the original model, to verify reconstruction.

\`\`\`python
from jev_hunt.attribution import compute_signed_contributions, select_evidence_frames

result = compute_signed_contributions(
    attention=attention,
    values=values,
    lr_weights=readout.weights.reshape(len(readout.head_indices), readout.head_dim),
    bias=readout.bias,
    token_frame_ids=token_to_frame,
    expected_logit=logit,  # raises if reconstruction fails
)
frame_ids = select_evidence_frames(result, budget=6)
\`\`\`

Do NOT simply read \`output_attentions=True\` from a different forward call and assume exact equivalence. Explicitly validate token mapping, padded tokens, RoPE, mask, query position and head order against the original frozen LR. If reconstruction fails, fall back to uniform samples, log the failure, and do not attribute gain to attention. Document runtime overhead.

## 4. Freeze InternVL3 and implement an EvidenceProvider

Every \`EvidenceItem\` must carry **a source frame ID and timestamp**, a declared claim ID, a concise textual observation, one of \`anomaly/normal/unknown\`, visibility and phase. An extractor may output unknown. The flag \`visible=True\` is a *model-reported visible fact*, not an external validation.

Create an adapter for the original frozen InternVL3, not a new model. Contract:

\`\`\`python
from jev_hunt.evidence import EvidenceItem, EvidenceRecord

class FrozenInternVLRescan:
    def rescan(self, record: EvidenceRecord, *, action: str, focus_id: str):
        # TODO: use the SAME frozen InternVL3 that supplies HeadHunt heads.
        # Map action to bounded frame/timestamp selection. Crop+full image for zoom.
        # Ask neutral, atomic questions with NO original anomaly score in prompt.
        # Construct EvidenceItem(..., phase="rescan") with real source metadata.
        raise NotImplementedError("Implement model-specific video access here")
\`\`\`

Recommended prompt structure: "At frame IDs [ids], describe ONLY directly visible actors, actions, contacts, object changes and occlusions. For the focal question [question], output whether it is observed true, observed false, or not visible; provide supporting frame IDs. Do not guess intent, unobserved violence or invisible objects."

**Potential evidence failures:** generated text can invent events, misattribute agents, conflate neighboring timestamps, or obey malicious text inside a frame. Human audits and adversarial probes are necessary.

## 5. Plug real evidence into the decision workflow

\`\`\`python
from jev_hunt.jev_client import JevClient
from jev_hunt.policy import CorrectionPolicy
from jev_hunt.workflow import ReviewWorkflow

client = JevClient(model="jev-latest")  # requires TYPESAFE_API_KEY at request time
try:
    outcome = ReviewWorkflow(
        decider=client,
        provider=FrozenInternVLRescan(),
        policy=CorrectionPolicy(kappa=0.35, max_logit_shift=1.0),
    ).run(original_logits, records_by_segment, review_fraction=0.10, seed=42)
    all_continuous_scores = outcome.to_dict()["corrected_scores"]
finally:
    client.close()
\`\`\`

Both Jev requests on a reviewed segment use the vendor's official \`Choice\` and \`Noul\` question types. Phase one chooses a focus and action, then **the actual frozen visual model** obtains more evidence; phase two evaluates the augmented evidence. Failures return the baseline logit.

Set key only in runtime environment: \`export TYPESAFE_API_KEY=...\`. For a stable research comparison pin a concrete Jev model version after checking your account and recording the vendor API/SDK versions. No API credentials, raw videos or private paths should go in manifests, logs, or git.

## 6. Demonstration using a cached manifest

\`\`\`bash
python -m pip install -e ".[dev]"
pytest -q
python -m jev_hunt.cli examples/sample_manifest.json --dry-run
# Optional real Jev API call using SYNTHETIC text; may incur usage charges:
python -m pip install -e ".[jev]"
export TYPESAFE_API_KEY="..."
python -m jev_hunt.cli examples/sample_manifest.json --live-jev --review-fraction 1
\`\`\`

In dry-run mode, the decision policy is a transparent no-op. In live mode, the only possible "rescans" come from a prewritten manifest. Thus **neither mode constitutes real new video observation**. Use this solely to validate API plumbing. A publishable experiment must implement section 4.

## 7. Reproducibility checklist

- HeadHunt commit/model revision, RHI seed and exact LR \`classes_\` ordering recorded.
- Paper vs public-repository discrepancies audited before running new experiments.
- Original logits and original \`predict_proba\` match to numerical tolerance.
- Attributions reconstruct exactly; unverified tokens never receive frame IDs.
- Jev question/options/model/version recorded; option permutations run as robustness controls.
- No labels or future frames leak into live prompts or online observations.
- Model score remains unchanged on every failed/inadequate review.
- All frames retained for primary AUC/AP; report extra compute/API errors/abstentions.
