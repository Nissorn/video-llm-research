# 5. Hidden-state and latent-space techniques for video LLMs

*Researched 2026-10-07 against arXiv papers, official repos and model cards. Related:
[§2](02-efficient-video-understanding.md) (the same goal via sampling/selection),
[§3](03-video-transformer-architectures.md) (V-JEPA as the pretraining-side expression).*

All six families intervene on model internals (attention maps, hidden states, logits, KV
cache), so **every method below requires open weights** unless noted. They split into
three purposes: cheaper inference (5.1, 5.5), more truthful answers (5.2, 5.6), and
better-calibrated/smarter reasoning (5.3, 5.4).

## 5.1 Visual-token pruning guided by internal signals

[FastV](https://arxiv.org/abs/2403.06764) (ECCV 2024 oral) observed that attention to
visual tokens collapses after ~layer 2; it prunes low-attention visual tokens in deep
layers, cutting ~45% FLOPs on LLaVA-1.5-13B with negligible loss (also validated on
Video-LLaVA). [VTW](https://arxiv.org/abs/2405.05803) (AAAI 2025) goes further: visual
tokens are only needed in the first half of LLM layers, after which hidden states suffice
— withdrawing them yields up to ~2x prefill and 4x decode speedup at full quality.
[SparseVLM](https://arxiv.org/abs/2410.04417) (ICML 2025) scores tokens via text-guided
self-attention and recycles them: 54% FLOPs cut, 37% lower latency, 97% accuracy kept.
[VisionZip](https://arxiv.org/abs/2412.04467) (CVPR 2025) keeps just 10% of tokens (~95%
performance, 8x prefill speedup). [MustDrop](https://arxiv.org/abs/2411.10525) drops
tokens across encoder/prefill/decode stages (~22.5% end-to-end speedup, strong on video),
and [FitPrune](https://arxiv.org/abs/2503.16287) fits pruning ratios from
attention-entropy statistics (~57.5% tokens pruned at +0.4 avg score). **Video-specific
2025 work is where the leverage is**: [HoliTom](https://arxiv.org/abs/2505.21334)
exploits global temporal redundancy (outer-LLM segmentation + inner-LLM merging) to reach
**6.9% of original FLOPs with 99.1% performance retained** on LLaVA-OneVision;
[PruneVid](https://arxiv.org/html/2412.16117v1) (ACL 2025),
[STTM](https://arxiv.org/html/2507.07990v1) (ICCV 2025), and query-guided
[LGTTP](https://aclanthology.org/anthology-files/anthology-files/pdf/emnlp/2025.emnlp-main.451.pdf)
(EMNLP 2025) merge/prune along both spatial and temporal axes; TimeChat-Online's DTD
drops 82.8% of redundant streaming tokens at 1.76x lower latency
([arXiv](https://arxiv.org/abs/2504.05277)). *Practicality: open-weights-only (needs
attention maps), but training-free and library-available (HuggingFace).*

## 5.2 Latent-space decoding fixes for multimodal hallucination

These are logit/hidden-state-level interventions at decode time.
[VCD](https://arxiv.org/abs/2311.16922) (CVPR 2024) contrasts logits from the original
vs. a distorted/noised visual input to suppress language priors;
[M3ID](https://arxiv.org/abs/2403.14003) (CVPR 2024) samples tokens maximizing mutual
information with the image — 25% fewer hallucinated objects and +21% POPE accuracy on
LLaVA-13B; [OPERA](https://arxiv.org/abs/2311.17911) (CVPR 2024) penalizes over-trust on
"summary tokens" with retrospective beam rollback;
[PAI](https://arxiv.org/abs/2407.21771) (ECCV 2024) amplifies image-token attention to
counter "text inertia"; [ICD](https://arxiv.org/abs/2405.13871) (ICML 2024) uses
intermediate-layer logits as contrastive priors; [DeGF](https://arxiv.org/abs/2406.08845)
(NeurIPS 2024) renders the draft answer as an image and feeds generative feedback into
both logits and hidden states. **Video transfer exists but is young**:
[VideoHallucer](https://arxiv.org/abs/2406.16336) is the standard video eval, and
[EventHallusion](https://arxiv.org/abs/2409.16597)'s Temporal Contrastive Decoding
contrasts against temporally disrupted videos to cut event hallucinations, with "evident
improvements" but no single headline number; 2025-26 successors (SEASON,
spatiotemporal-semantic CD) are preprints/very recent. *Practicality: open-weights-only;
VCD-style methods double decode cost (second forward pass), and text-quality side effects
are documented in 2025 critiques.*

## 5.3 Hidden-state probing for calibration / hallucination detection

[SAPLMA](https://arxiv.org/abs/2304.13734) (Azaria & Mitchell, EMNLP-Findings 2023)
trains an MLP on hidden states to classify whether the model's own statement is true,
beating self-assessment prompting; [LLM Factoscope](https://arxiv.org/abs/2312.05107)
(ACL-Findings 2024) probes single-layer states for factual discernment;
[INSIDE](https://arxiv.org/abs/2402.03744) (ICLR 2024) detects hallucination via
EigenScore over the covariance of internal states of sampled answers;
[Semantic Entropy Probes](https://arxiv.org/abs/2406.15927) (ICLR 2025) show a linear
probe on **one** hidden state approximates costly semantic-entropy sampling. Video-side:
[VideoICL](https://arxiv.org/abs/2412.02186) (CVPR 2025) trains an MLP probe on a
video-LLM's hidden states to score answer confidence and select informative demos for OOD
video QA; 2025-26 work (UniProbe, EnsemHalDet, ICR-Probe) extends token-level detection
for LVLMs (survey:
[Awesome-LLM-LVLM-Hallucination-Detection](https://github.com/mala-lab/Awesome-LLM-LVLM-Hallucination-Detection-and-Mitigation)).
So yes — a probe can plausibly gate "this answer is unreliable, fetch more frames."
*Practicality: open-weights-only and needs labeled probe data, but the probe itself is
cheap at runtime; for API models only output-level proxies (self-consistency, logprobs)
work.*

## 5.4 Latent/continuous reasoning

[Coconut](https://arxiv.org/abs/2412.06769) (Hao et al., ICLR 2025) feeds the last hidden
state back as a "continuous thought" instead of decoding tokens, beating CoT on
planning-heavy tasks and encoding multiple candidate paths in one latent thought.
Multimodal successors: [Mirage](https://arxiv.org/abs/2506.17218) (CVPR 2026)
interleaves latent visual tokens in decoding — 58% vs. 47% for CoT on VSP spatial
reasoning; [Latent Visual Reasoning](https://arxiv.org/abs/2509.24251) (2025) generates
latent states that reconstruct needed visual tokens (MMVP 71.67% vs. 66.67% for
Qwen2.5-VL, image-only); [Visual Perception Tokens](https://arxiv.org/abs/2502.17425) let
an MLLM re-attend to the image mid-reasoning (~15% gains; see also
[Aurora](https://aurora-perception.github.io)). None yet show video results. The
pretraining-side expression of the same philosophy is
[V-JEPA 2](https://arxiv.org/abs/2506.09985) (Meta, 2025): prediction purely in
representation space over 1M+ hours of video (SSv2 77.3, PerceptionTest 84.0, zero-shot
robot planning), with no pixel decoding. *Practicality: research-only — requires custom
training; no off-the-shelf video model.*

## 5.5 KV-cache tricks for video

[ReKV](https://arxiv.org/abs/2503.00540) (ICLR 2025) is the canonical "cache once, answer
many": it stream-encodes video with sliding-window attention, offloads KV to RAM/disk,
and retrieves only query-relevant KV segments — training-free, supports hours-long
videos, and can use either the model's internal attention (white-box) or an external
retriever. General compressors transfer to video:
[SnapKV](https://arxiv.org/abs/2404.14469) (NeurIPS 2024; 3.6x faster generation, 8.2x
memory) and [PyramidKV](https://arxiv.org/abs/2406.02069) (pyramidal per-layer budgets;
matches full-cache with ~12% of the cache, benchmarked on Video-MME with
LLaVA-Video-7B). 2025-26 video-specific successors include
[InfiniPot-V](https://arxiv.org/html/2506.15745v2) (on-device streaming), MotionKV
(motion-aware retrieval), and StreamKV (AAAI 2026). *Practicality: open-weights-only, but
well-supported in HuggingFace tooling; the strongest architectural fit for a
follow-up-question CLI.*

## 5.6 Representation engineering / steering

RepE ([Zou et al.](https://arxiv.org/abs/2310.01405), 2023) steers behavior via reading
vectors; multimodally, [VTI](https://arxiv.org/abs/2410.15778) (Liu et al., 2024)
stabilizes unstable vision-encoder features (TopK/SVD) and steers LLM latents toward
image-grounded tokens, cutting POPE hallucinations without training; SHARP (ACL 2025) and
DMAS (ICLR 2026) add decomposed/dynamic steering vectors for LVLM hallucination. Gains
are modest, direction calibration is fragile, and there is **no video-specific evidence**.
*Practicality: open-weights-only; research-only.*

## Verdict for a transcript-first video QA CLI

1. **Adopt: ReKV-style "cache once, answer many" KV retrieval** (5.5) — if the CLI runs
   an open-weights video LLM, encoding the video once and retrieving per follow-up is the
   single biggest latency/cost win, with an open implementation
   ([GitHub](https://github.com/Becomebright/ReKV)).
2. **Adopt: video token merging** (5.1) — HoliTom/PruneVid-style temporal merging (FLOPs
   to ~7% at ~99% quality) is training-free and directly targets frame-heavy inference;
   relevant whenever the CLI sends frames to an open-weights VLM.
3. **Adopt (lightweight): a confidence probe as a "fetch more frames" gate** (5.3) — a
   VideoICL/SEP-style linear probe on hidden states is cheap and answers "is this
   transcript-grounded answer reliable?"; for API-only models, fall back to
   self-consistency voting.
4. **Research-only**: contrastive-decoding fixes (5.2) work but cost extra passes and
   mainly fix object hallucination on frames, not transcript-grounded QA; latent visual
   reasoning (5.4) has no video results and needs training; steering (5.6) is fragile and
   unproven on video.
