# Video Understanding for LLMs — Research Notes

Research compiled **2026-10-07** on how LLMs ingest and understand video, what it costs
in tokens, and which techniques make it cheaper and better — plus adjacent model-class
surveys: classic video transformers, "System One" decision models (TypeSafe's **Jev** and
its clone ecosystem), and hidden-state / latent-space methods.

**Method.** Sections were researched by parallel background research agents against
**primary sources** — official API docs, arXiv papers, official repos and model cards.
Every claim links to the source that owns it; **⚠️ marks items taken from secondary
sources that were not independently opened/verified** (mostly in section 4).

Originally compiled as a single document while working on a YouTube "ask the video" CLI
([askyt](https://github.com/Nissorn)); split into per-topic files here since the topic
stands on its own.

## Contents

Cross-references in the text use `§N`, matching the file numbers below.

| § | File | What's in it |
|---|---|---|
| 1 | [01-frontier-video-ingestion-and-cost.md](01-frontier-video-ingestion-and-cost.md) | How Gemini / OpenAI / Claude / Qwen actually ingest video, and tokens-per-minute arithmetic per provider |
| 2 | [02-efficient-video-understanding.md](02-efficient-video-understanding.md) | Query-aware keyframe selection, token compression, streaming memory, transcript-first results, practical recipe |
| 3 | [03-video-transformer-architectures.md](03-video-transformer-architectures.md) | TimeSformer / ViViT / VideoMAE / InternVideo / V-JEPA / VideoMamba — classification-only or more? |
| 4 | [04-jev-system-one-models.md](04-jev-system-one-models.md) | Jev (TypeSafe AI), the "System One" model class, its clones (Lichen, openjev, simple-jev…), diffusion-LLM neighbors, and what fast typed decisions could do in a video pipeline |
| 5 | [05-hidden-state-latent-space-techniques.md](05-hidden-state-latent-space-techniques.md) | Attention-guided token pruning, contrastive decoding, hallucination probes, latent reasoning, KV-cache tricks, steering — adopt vs research-only |
| 6 | [06-case-study-video-qa-cli.md](06-case-study-video-qa-cli.md) | Case study: applying all of the above to a transcript-first video QA CLI |

## Headline numbers

- Native video input prices differ ~10x across providers: Gemini ≈ **5,700 tokens/min**
  (default) vs frame-sampled GPT-4o ≈ **66,000 tokens/min** at 1 fps high detail (§1).
- On long, speech-heavy video, **adding subtitles is worth +10.1%** (Video-MME) —
  transcript-first is the biggest single lever (§2).
- Query-aware frame selection reaches SOTA-level video QA with **~8 frames** instead of
  ~180 (VideoAgent) (§2).
- Video token merging can cut inference to **~7% of FLOPs at ~99% quality** (HoliTom) (§5).
- Chat-capable video LLMs all use **per-frame image ViTs** (SigLIP etc.), not video-native
  spatiotemporal encoders — those survive as cheap pipeline components (§3).

## License

MIT — see [LICENSE](LICENSE).
