# 6. Case study — applying this to a transcript-first video QA CLI

*Context: this section applies the research to **askyt**, the real CLI project this
research started from — a tool that asks questions about a YouTube video. Its current
path delegates all video understanding to YouTube's own "Ask about this video" backend
(YouChat/Gemini), reverse-engineered in the
[askyt repo](https://github.com/Nissorn). Related: [§1](01-frontier-video-ingestion-and-cost.md),
[§2](02-efficient-video-understanding.md), [§4](04-jev-system-one-models.md).*

askyt currently delegates all video understanding to YouTube's YouChat backend: we send
only the question text, and Google's server does retrieval over the video. That means
**none of the cost/quality levers in [§2](02-efficient-video-understanding.md) are under
our control** — we pay zero tokens, but we also can't tune sampling, inject our own
context, or guarantee which evidence the model sees. Practical options, cheapest first:

1. **Stay on YouChat for the default path** (zero tokens, timestamp citations for free),
   and treat it as the baseline other modes must beat.
2. **Gemini API with the YouTube URL**
   ([video-understanding docs](https://ai.google.dev/gemini-api/docs/video-understanding))
   — server-side ingest, ~5,700 tok/min default; a 10-min video ≈ 57K tokens ≈ $0.02 on
   2.5 Flash. Gives us control (system prompt, structured output with our own schema,
   multi-turn) at trivial cost for short videos.
3. **Transcript-first + query-aware keyframes**
   ([§2.4 + §2.1](02-efficient-video-understanding.md)): YouTube captions/ASR are free
   and, per Video-MME, worth ~+10% on long speech-heavy videos; add 8–16 CLIP/SigLIP
   -selected frames only for visually grounded questions. This is the best
   tokens-per-quality ratio and the only path that also works fully offline of Google's
   ask backend.

Structured/type-safe output (JSON schema, Zod-style parsing) is orthogonal: it guarantees
the *shape* of answers (timestamps, quotes, confidence) but does not improve what the
model *perceives* — perception quality is decided by which evidence (transcript, frames,
audio) reaches the context window. A Jev-class decision model
([§4](04-jev-system-one-models.md)) is the natural "smart if-statement" for the routing
layer itself (transcript-only vs fetch-frames vs re-answer), sitting on top of that
evidence selection rather than replacing it. If the pipeline ever moves to open weights,
the [§5](05-hidden-state-latent-space-techniques.md) levers (ReKV caching, token merging,
confidence probes) become available on top.
