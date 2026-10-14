# Research bibliography and novelty audit (60 publications)

**Snapshot:** 2026-10-14. This is a **working literature inventory**, not 60 end-to-end experimental reproductions or a claim every paper was reviewed line-by-line. Before publication re-check versions, dates, author lists, peer-review status, bib metadata, and whether newer work supersedes these preprints. References alone do not prove novelty.

HeadHunt's attention-head readout, Jev as text-only Choice/Noul, and recent work on proposition-driven evidence escalation need to be discussed together. In particular, the direct Jev-VAD study (#21) prevents any claim that merely using Jev for VAD is first-of-kind. PARSEE-VAD (#16), QVAD (#17), AgenticVAU (#18) and MACD (#19) prevent claims that evidence escalation and active inspection are new in isolation.

**Our narrow research hypothesis:** map the **original frozen linear head-readout's signed token contributions** to candidate visual observations, then test whether existing Jev choices improve *which additional evidence to obtain*, at equal visual and latency budgets. See [METHOD](METHOD.md) and [EXPERIMENTS](EXPERIMENTS.md).


## A. Baseline, training-free and weakly-supervised VAD (1–15)

1. [HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection](https://arxiv.org/abs/2512.17601) — Frozen detector and selected-head LR being extended.
2. [HiProbe-VAD: Video Anomaly Detection via Hidden States Probing in Tuning-Free Multimodal LLMs](https://arxiv.org/abs/2507.17394) — Alternative probing location; do not confuse probing gains with Jev gains.
3. [Harnessing Large Language Models for Training-free Video Anomaly Detection (LAVAD)](https://arxiv.org/abs/2404.01014) — Description-mediated VAD and its textual bottleneck.
4. [VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models](https://arxiv.org/abs/2412.01095) — Question-driven VAD; distinguish trained verbalized adaptation.
5. [EventVAD: Training-Free Event-Aware Video Anomaly Detection](https://arxiv.org/abs/2504.13092) — Event-based prompting is established prior art.
6. [VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection](https://arxiv.org/abs/2308.11681) — Task-specific anomalous text semantics with adaptation.
7. [Video Anomaly Detection and Explanation via Large Language Models (VAD-LLaMA)](https://arxiv.org/abs/2401.05702) — Joint detection/explanation requires separate training.
8. [Holmes-VAD: Towards Unbiased and Explainable Video Anomaly Detection via Multi-modal LLM](https://arxiv.org/abs/2406.12235) — Instruction-data VAD; not a frozen inference-only analogue.
9. [Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning (RTFM)](https://arxiv.org/abs/2101.10030) — Temporal sparsity and video-level supervision constraints.
10. [Dual Memory Units with Uncertainty Regulation for Weakly Supervised Video Anomaly Detection](https://arxiv.org/abs/2302.05160) — Normal/anomaly memory and uncertainty; excludes new learned memory.
11. [MIST: Multiple Instance Self-Training Framework for Video Anomaly Detection](https://arxiv.org/abs/2104.01633) — Local snippet vs video-level label mismatch.
12. [Real-world Anomaly Detection in Surveillance Videos (UCF-Crime)](https://arxiv.org/abs/1801.04264) — Standard benchmark and evaluation.
13. [Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision (XD-Violence)](https://arxiv.org/abs/2007.04687) — Audio modality control and XD-Violence benchmark.
14. [UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection](https://arxiv.org/abs/2111.08644) — Open-set generalization stress test.
15. [Street Scene: A new dataset and evaluation protocol for video anomaly detection](https://arxiv.org/abs/1902.05872) — Event-level localization and metric audit.

## B. Most overlapping 2026 evidence-routing systems (16–20)

16. [PARSEE-VAD: Efficient Training-Free Online Video Anomaly Detection via Proposition-Aware Reasoning and Streaming Evidence Escalation](https://arxiv.org/abs/2609.33236) — Selective proposition review, evidence escalation, bounded score updates already exist.
17. [QVAD: A Question-Centric Agentic Framework for Efficient and Training-Free Video Anomaly Detection](https://arxiv.org/abs/2604.03040) — Question-driven rescan; novelty requires LR-aligned evidence choice.
18. [AgenticVAU: Multi-Agent Explore-Verify Reasoning for Video Anomaly Understanding](https://arxiv.org/abs/2608.03779) — Explore/verify loop and targeted observation are prior work.
19. [MACD: Online Video Anomaly Detection via Multi-Agent Counterfactual Dialectics](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/sfw2/6947479) — Normal explanation and visual challenge precedents.
20. [Parser-Free VLM Verification for Federated Weakly Supervised Video Anomaly Detection](https://arxiv.org/abs/2609.07455) — Strong frozen-VLM binary-likelihood reviewer baseline.

## C. Jev-specific and direct Jev/VAD findings (21–25)

21. [Decision Readouts for Text-Mediated Video Anomaly Detection: An Exploratory Evaluation of Jev and Qwen](https://arxiv.org/abs/2609.34180) — Direct prior Jev video-anomaly readout comparison, mixed outcomes.
22. [NumericJev: Jev-like LLM Numerical Decoding with Multiway Decision Trees](https://arxiv.org/abs/2609.28587) — Jev-like is not the official Jev model; numerical interface caveat.
23. [Jev for Scientific Decisions: Evaluating Semantic Choices and Their Consequences](https://arxiv.org/abs/2609.24965) — Separation of semantic choice from deterministic downstream calculations.
24. [HydroJEV: A one-second, training-free screen for cyber-attack and fault attribution in water distribution networks](https://arxiv.org/abs/2610.02048) — Decision cascade with rules, but domain-specific safety/speed claims do not transfer.
25. [A First Glance at Jev for Network Traffic Classification: Accuracy, Processing Time, and Cost](https://arxiv.org/abs/2610.00376) — Domain and supervision mismatch risks for Jev numerical classification.

## D. Heads, attribution and internal representations (26–34)

26. [Are Sixteen Heads Really Better than One?](https://proceedings.neurips.cc/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html) — Head importance varies; frozen LR readout is not network pruning.
27. [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://aclanthology.org/P19-1580/) — Head specialization is task-dependent, not a causal anomaly localization proof.
28. [Attention is not Explanation](https://aclanthology.org/N19-1357/) — Attention weights alone cannot be claimed causal.
29. [Attention is not not Explanation](https://arxiv.org/abs/1908.04626) — Interpretability depends on appropriate counterfactual controls.
30. [Attention is Not Only a Weight: Analyzing Transformers with Vector Norms](https://arxiv.org/abs/2004.10102) — Motivates value-aware contributions beyond alpha-only saliency.
31. [Measuring the Mixing of Contextual Information in the Transformer](https://aclanthology.org/2022.emnlp-main.595/) — Deep positions contain mixed context and do not cleanly localize pixels.
32. [Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/) — Information-flow alternatives to simple head attention.
33. [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/abs/2306.03341) — Typical intervention needs trained directions; avoid calling this purely tuning-free.
34. [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827) — Unsupervised probe learning is distinct from no training.

## E. Hallucination, temporal grounding, atomic facts (35–44)

35. [Evaluating Object Hallucination in Large Vision-Language Models (POPE)](https://aclanthology.org/2023.emnlp-main.20/) — Atomic visible-object claims require checking.
36. [HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2024/html/Guan_HallusionBench_An_Advanced_Diagnostic_Suite_for_Entangled_Language_Hallucination_and_CVPR_2024_paper.html) — Visual prior vs evidence conflict.
37. [VideoHallucer: Evaluating Intrinsic and Extrinsic Hallucinations in Large Video-Language Models](https://arxiv.org/abs/2406.16338) — Frame-order and object hallucination controls.
38. [Taming Object Hallucinations with Verified Atomic Confidence Estimation (TACO)](https://arxiv.org/abs/2511.09228) — Atomic fact verification as earlier research, not separate novelty.
39. [GEASS: Gated Evidence-Adaptive Selective Caption Trust for Vision-Language Models](https://arxiv.org/abs/2605.01733) — Evidence gating before Jev-Hunt; use as a comparator.
40. [Beneath the Scores: Rethinking Hallucination Evaluation for Video Understanding Models](https://arxiv.org/abs/2609.28991) — Separate perception, localization, and reasoning failures.
41. [Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://arxiv.org/abs/2311.16922) — Training-free visual contrast; artificial perturbations are not ground truth.
42. [OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation](https://arxiv.org/abs/2311.17911) — Generation attention/over-trust artifacts.
43. [TempCompass: Do Video LLMs Really Understand Videos?](https://arxiv.org/abs/2403.00476) — Temporal contrast cases and static-frame shortcuts.
44. [MVBench: A Comprehensive Multi-modal Video Understanding Benchmark](https://arxiv.org/abs/2311.17005) — Motion/sequence reasoning diagnostics.

## F. Calibration, selection and risk (45–52)

45. [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) — Sigmoid/Noul scores need held-out calibration before probability semantics.
46. [Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift](https://papers.nips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift) — Uncertainty confidence fails under shift.
47. [Selective Classification for Deep Neural Networks](https://papers.nips.cc/paper/7073-selective-classification-for-deep-neural-networks) — Coverage-risk must accompany abstention.
48. [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221) — Truth-assessment readouts require calibration and task checks.
49. [Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation](https://arxiv.org/abs/2302.09664) — Meaningful uncertainty requires semantic equivalence.
50. [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0) — Agreement may coexist with systematic hallucination.
51. [Conformal Risk Control](https://openreview.net/forum?id=33XGfHLtZg) — Risk bounds need calibration data and assumptions.
52. [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170) — Online distribution shift requires explicit feedback assumptions.

## G. Frozen VLMs, local observation and budget (53–60)

53. [InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models](https://arxiv.org/abs/2504.10479) — Freeze exact backbone and vision processor revision.
54. [WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation](https://arxiv.org/abs/2303.14814) — Local/global visual anomaly complementarity, mostly images.
55. [AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection](https://arxiv.org/abs/2310.18961) — Anomaly vocabulary vs object semantics; trained prompts differ from no-training.
56. [AdaFrame: Adaptive Frame Selection for Fast Video Recognition](https://arxiv.org/abs/1811.12432) — Learned adaptive sampling is historical prior.
57. [Flexible Frame Selection for Efficient Video Reasoning](https://openaccess.thecvf.com/content/CVPR2025/html/Buch_Flexible_Frame_Selection_for_Efficient_Video_Reasoning_CVPR_2025_paper.html) — Frame budget matched dynamic selection.
58. [MDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs](https://arxiv.org/abs/2501.02885) — Strong training-free diversity-based frame-selection baseline.
59. [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176) — Cost-aware cascades need matched budget; learned routers not included.
60. [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://proceedings.mlr.press/v139/radford21a.html) — General frozen visual-language backbone; do not silently add another model.

---

## Implications for claim writing

1. **Do not claim** training-free HeadHunt + Jev readout is new by itself (#1, #16–#21).
2. **Do not claim** confidence/uncertainty gating is new by itself (#16, #39, #45–#52).
3. **Do not claim** attention weights locate causes; validate precise LR reconstruction and retrieval quality (#26–#34).
4. **Do not claim** rescanning fixes hallucinations automatically: both original and rescan summaries come from the same frozen InternVL3 (#35–#44).
5. **Do not claim** faster online inference without total GPU, IO, text token and API timing; include strict causal deadlines (#53–#59).
6. The experimental hypothesis remains falsifiable against signed-vs-attention-only frame selection and Jev-vs-frozen-InternVL3 action choice.

For Jev model capabilities and response contracts (not counted among papers), consult first-party documentation:
- https://docs.typesafe.ai/concepts/system-one
- https://docs.typesafe.ai/primitives/choice
- https://docs.typesafe.ai/primitives/noul
