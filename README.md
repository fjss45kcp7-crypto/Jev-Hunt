# Jev-Hunt

**Head-grounded Evidence Acquisition and Selective Decisions for Frozen Video Anomaly Detection**

Jev-Hunt is an experimental, **training-free extension of an existing HeadHunt-VAD detector**. It does **not** train Jev, fine-tune InternVL3, or replace HeadHunt-VAD's selected attention heads or logistic regression scorer. It investigates whether internal, signed anomaly-readout contributions can guide additional visual evidence collection and whether TypeSafe's text-only **Jev** can select useful follow-up observations.

> **Status: research prototype.** The attribution, review-budget, Jev typed-choice adapter, conservative correction policy, cached-evidence workflow, and offline unit tests are implemented. End-to-end raw-video extraction / InternVL3 rescanning is intentionally provided as an integration contract, **not** claimed to be working out of the box. No benchmark improvement is claimed.

## Pipeline

```text
Existing frozen HeadHunt-VAD -> original logits + head features
                     |                   |
                     +-> signed head/token evidence proposal
                                          |
             budgeted segment selection + low-score audits
                                          |
                   timestamped visual facts (same InternVL3)
                                          |
                      Jev Choice / Noul decision
                                          |
             one targeted rescan with frozen InternVL3
                                          |
                    Jev evidence re-evaluation
                                          |
             conservative gated logit correction (optional)
```

Jev receives **text / JSON evidence only**, never images or attention vectors. A Jev score is **not** assumed calibrated on video anomaly detection. Unresolved and failed calls leave original logits intact.

## Install and smoke test

Requires Python >=3.10.

```bash
git clone https://github.com/fjss45kcp7-crypto/Jev-Hunt.git
cd Jev-Hunt
python -m pip install -e ".[dev]"
pytest -q
python -m jev_hunt.cli examples/sample_manifest.json --dry-run
```

Use live Jev **only** with an existing TypeSafe account and API key:

```bash
python -m pip install -e ".[jev]"
export TYPESAFE_API_KEY="..."  # never commit this or your output logs
python -m jev_hunt.cli examples/sample_manifest.json --live-jev
```

The sample manifest contains **invented synthetic evidence for interface testing**, not validated detections or research results. Real experiments must pass HeadHunt logits and fresh, timestamp-grounded InternVL3 observations through the interfaces documented below. The CLI's dry run intentionally does not simulate Jev accuracy.

## Research documents

- [Method and mathematical assumptions](docs/METHOD.md)
- [Implementation and frozen-model integration](docs/INTEGRATION.md)
- [Experiments, baselines and leakage checks](docs/EXPERIMENTS.md)
- [60-paper related-work inventory](docs/RELATED_WORK.md)

## Research questions

1. Does **signed readout contribution** select more decisive frames than uniform, random, attention-only and training-free diverse sampling under equal budgets?
2. Does Jev choose missing evidence / rescan actions better than fixed heuristics or the same frozen MLLM's structured readout?
3. Does gated correction improve full-coverage ROC-AUC/AP **without disguising coverage, risk or compute cost**?

There is significant related prior art for selective reasoning, rescan and Jev VAD readouts. We do **not** claim to invent those ingredients individually. See the overlap and falsifiable novelty claims in [METHOD.md](docs/METHOD.md).

## Attribution caveat

For selected head `k`, query position `q` and input key position `j`:

```
r[k,j] = attention[k,q,j] * dot(lr_weight[k], value[k,j])
original_logit = lr_bias + sum(r[k,j])    # only if extraction matches exact model computation
```

The result is a *readout contribution under the current forward pass*, **not** a causal anomaly explanation. Always verify the reconstruction residual before treating it as valid; mapped frames exclude text tokens. Do not reweight the original LR inputs.

## Credits and responsible claims

Based on **HeadHunt-VAD** (Cai et al., AAAI 2026; [paper](https://arxiv.org/abs/2512.17601), [code](https://github.com/CebCai/HeadHunt-VAD)). Jev is TypeSafe's hosted, text-only decision model; [official docs](https://docs.typesafe.ai/concepts/system-one). This repository does not distribute InternVL3 or HeadHunt-VAD weights and does not embed API credentials.

**Not a safety-critical monitoring system.** Do not use prototype decisions as autonomous evidence of wrongdoing.
