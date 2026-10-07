# 3. Video transformer architectures beyond classification

*Researched 2026-10-07 against arXiv papers, HF model pages and official repos.*

**Short answer:** classic video-native transformers (TimeSformer/ViViT/VideoMAE/Video
Swin) are stuck as classification/detection backbones, video-text contrastive models
genuinely power retrieval and embeddings, JEPA-style world models opened
planning/robotics, and — notably — none of the spatiotemporal encoders made it into
chat-capable video LLMs, which all still use image ViTs per-frame.

| Family | Representative model (venue/year) | Main tasks today | Inside chat VLMs today? | Usable as component in an LLM pipeline? |
|---|---|---|---|---|
| Video-native encoders | [TimeSformer](https://arxiv.org/abs/2102.05095) (ICML 2021, divided space-time attention), [ViViT](https://arxiv.org/abs/2103.15691) (ICCV 2021, factorized attention/tubelets), [VideoMAE](https://arxiv.org/abs/2203.12602) (NeurIPS 2022) / [VideoMAE V2](https://arxiv.org/abs/2303.16727) (CVPR 2023, dual masking, 1B-param ViT-g), [Video Swin](https://arxiv.org/abs/2106.13230) (CVPR 2022), [MVD](https://arxiv.org/abs/2212.04548) (CVPR 2023) | Action classification (K400 ~90%, SSv2), action detection (AVA) | **No** | Limited — frozen feature extractors for probe/detection heads |
| Video-text contrastive / retrieval | [CLIP4Clip](https://arxiv.org/abs/2104.08860) (arXiv 2021), [X-CLIP](https://arxiv.org/abs/2208.02816) (ECCV 2022 oral), [VideoCLIP](https://arxiv.org/abs/2109.14084) (EMNLP 2021), [InternVideo](https://arxiv.org/abs/2212.03191) / [InternVideo2](https://arxiv.org/abs/2403.15377) (ECCV 2024) | Zero-shot recognition, **text-to-video retrieval**, captioning/dialogue (InternVideo2) | Rarely ([VideoChat-Flash](https://arxiv.org/abs/2501.00574) uses InternVideo2-CLIP-1B) | **Yes** — embeddings for video search / moment retrieval / reranking |
| JEPA / world models | [V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) (Feb 2024), [V-JEPA 2 / 2-AC](https://arxiv.org/abs/2506.09985) (2025) | Motion understanding (77.3 SSv2 frozen), action anticipation, zero-shot robot planning | No | **Potentially** — [Apache-2.0 weights](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256), in `transformers` |
| State-space / hybrid | [VideoMamba](https://arxiv.org/abs/2403.06977) (ECCV 2024) | Short/long-term action recognition | No | No — near-zero ecosystem uptake |
| Image ViT per-frame (baseline reality) | SigLIP, InternViT, EVA-ViT + pooling/Q-Formers | Chat VLM vision backbones | **Yes — this is what everyone uses** | **Yes** — frame scoring, dedup, retrieval |

## Video-native encoders: classification and little else

TimeSformer's
[divided space-time attention](https://proceedings.mlr.press/v139/bertasius21a/bertasius21a.pdf)
and ViViT's factorized variants were built for, and are still evaluated on,
Kinetics/SSv2-style action recognition ([ViViT abstract](https://arxiv.org/abs/2103.15691));
VideoMAE (masked tube pretraining, 90–95% masking) reports 87.4% K400 / 75.4% SSv2 /
91.3% UCF101 classification ([abstract](https://arxiv.org/abs/2203.12602)), and VideoMAE
V2 pushes a [1B-param ViT-g](https://arxiv.org/abs/2303.16727) with dual masking. Their
real-world footprint is classification serving:
[MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base) (~354k downloads),
[google/vivit-b-16x2-kinetics400](https://huggingface.co/google/vivit-b-16x2-kinetics400)
(~54k), [facebook/timesformer-base-finetuned-k600](https://huggingface.co/facebook/timesformer-base-finetuned-k600)
(~15k) — all tagged `video-classification`. **None of them appear inside modern chat
VLMs.** LLaVA-Video "employed SigLIP as our vision encoder" with per-frame encoding +
spatial average pooling ([§A.1](https://arxiv.org/html/2410.02713v3)); Qwen2.5-VL trains
"the redesigned ViT from scratch" on DataComp with windowed attention, frame pairs, and
absolute-time M-RoPE ([architecture §](https://arxiv.org/html/2502.13923));
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) keeps a ViT and adds DeepStack multi-level
features, Interleaved-MRoPE, and text–timestamp alignment
([report](https://arxiv.org/abs/2511.21631)); InternVL3 uses InternViT-300M/6B with pixel
unshuffle and 16–64 sampled frames ([report](https://arxiv.org/html/2504.10479v1));
[VideoLLaMA3](https://arxiv.org/abs/2501.13106) is explicitly vision-centric on image-text
data; [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID) uses EVA-ViT-G +
an InstructBLIP Q-Former (2 tokens/frame). The one notable counterexample is OpenGVLab's
own lineage: [VideoChat-Flash](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B)
swaps in InternVideo2-CLIP-1B.

## Where they genuinely win beyond classification: retrieval, anticipation, planning

Video-text contrastive models are deployed *as embeddings*, not classifiers. InternVideo2
(ECCV 2024) unifies masked video modeling, contrastive learning, and next-token prediction
over "60+ video and audio tasks" including retrieval and dialogue
([abstract](https://arxiv.org/abs/2403.15377)); CLIP4Clip remains the standard
text-to-video-retrieval baseline on MSR-VTT/DiDeMo ([paper](https://arxiv.org/abs/2104.08860)),
and [X-CLIP's](https://arxiv.org/abs/2208.02816)
[zero-shot checkpoint](https://huggingface.co/microsoft/xclip-base-patch16-zero-shot)
(~186k downloads, MIT) shows heavy practical use as an open-vocabulary video classifier.
This pattern extends to production: Twelve Labs' Marengo Embed 3.0 ships
[video-native embeddings on AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-multiemb-models.html)
for semantic video search, RAG, and recommendations. The JEPA line is the clearest
"beyond classification" story: V-JEPA predicts in latent space without pixel decoding,
and Meta's argument is that generative models waste capacity on unpredictable pixel
detail, while JEPA "has the flexibility to discard unpredictable information" for 1.5–6x
better efficiency ([Meta blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)).
V-JEPA 2 (1M+ hours pretraining) reaches 77.3 top-1 SSv2 frozen, SOTA 39.9 R@5
Epic-Kitchens action anticipation, and zero-shot Franka pick-and-place via latent
planning with V-JEPA 2-AC trained on <62h of Droid video ([paper](https://arxiv.org/abs/2506.09985));
weights are [open, Apache-2.0](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256)
(~95k downloads on the ViT-g checkpoint). By contrast, VideoMamba (ECCV 2024) is
competitive — ~82.2 SSv2, linear complexity, strong long-video results
([ECCV listing](https://eccv.ecva.net/virtual/2024/poster/341)) — but shows a known
overfitting weakness on Kinetics-scale data and has essentially
[no ecosystem uptake](https://huggingface.co/OpenGVLab/VideoMamba) (~0 downloads); it
beat transformers on paper, not in practice.

## Are they usable as cheap components in an LLM pipeline? Yes, selectively

The proven pattern is frozen, small, task-specific models around the LLM: (1) *frame
scoring/keyframe selection* — [VideoTree](https://arxiv.org/abs/2405.19209) (CVPR 2025)
extracts per-frame features with EVA-CLIP-8B to pick query-relevant keyframes before the
LLM, and [AKS](https://arxiv.org/abs/2502.21271) (CVPR 2025) optimizes keyframe
relevance+coverage; (2) *shot/scene detection* —
[TransNetV2](https://github.com/soCzech/TransNetV2) (~1,050 stars) is the de-facto
shot-boundary transformer for preprocessing; (3) *retrieval/moment localization* —
CLIP4Clip-style dual encoders or InternVideo2-CLIP as rerankers/index for video RAG; (4)
*anticipation/embeddings* — V-JEPA 2 features for physical-world signals a per-frame ViT
can't capture.

**Verdict:** classic video transformers are not classification-only — retrieval
embeddings, anticipation, world modeling, and pipeline tooling are real 2025/2026 uses —
but their *architectural* ideas (temporal attention, masked latent prediction) mostly lost
the chat-VLM backbone slot to image ViTs applied per-frame, surviving inside the LLM stack
as cheap auxiliary components rather than as the main vision encoder.
