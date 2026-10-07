# 2. Making video understanding cheaper and better

*Researched 2026-10-07 against arXiv papers and official repos. Related:
[§1](01-frontier-video-ingestion-and-cost.md) (cost baseline),
[§5](05-hidden-state-latent-space-techniques.md) (model-internal variants).*

Video cost is `frames × tokens-per-frame`, and naive uniform sampling is wasteful on both
axes: a 1-hour video at Gemini's default 1 fps setting costs ~360K tokens (258
tokens/frame, ~100 tokens/second) ([Gemini docs](https://ai.google.dev/gemini-api/docs/video-understanding)).
Four technique families attack this, and the evidence says they compose: cheap transcript
first, query-aware frames second, compressed tokens third.

## 2.1 Query-aware keyframe selection (fewer, better frames)

The core idea: score frames against the query instead of sampling uniformly.
**VideoAgent** (Wang et al., [arXiv 2024](https://arxiv.org/abs/2403.10517)) makes an LLM
agent iteratively decide which moments to inspect, using CLIP as a cheap tool to locate
query-relevant candidates; it answers EgoSchema at 54.1% and NExT-QA at 71.3% zero-shot
with only **8.4/8.2 frames on average** (~20x fewer than the 180-frame caption-aggregation
baseline LLoVi), and measured CLIP scoring at **<1% of total compute**
([paper](https://arxiv.org/abs/2403.10517)). **VideoTree**
([CVPR 2025](https://arxiv.org/abs/2405.19209)) builds a coarse-to-fine query-adaptive
tree over the video, reaching 61.1% EgoSchema / 75.6% NExT-QA training-free while beating
GPT-4V on Video-MME's long split. **AKS**
([CVPR 2025](https://arxiv.org/abs/2502.21271), [repo](https://github.com/nctimtang/AKS))
is a plug-and-play sampler optimizing both query relevance and coverage over the video: on
LLaVA-Video-7B it lifts LongVideoBench 58.9→62.7 (+3.8) and Video-MME 64.4→65.3 (+0.9)
over uniform sampling — big gains on weaker models (e.g., +5.0 LongVideoBench on
Qwen2-VL), smaller on strong ones. **Frame-Voyager**
([ICLR 2025](https://arxiv.org/abs/2410.03226)) trains a ranker to pick query-conditioned
frame *combinations* by distilling a Video-LLM's prediction loss. **Q-Frame**
([ICCV 2025](https://arxiv.org/abs/2506.22139)) adds multi-resolution scaling — more
tokens for the most query-relevant frames — as a training-free, CLIP-scored,
Gumbel-Max-selected scheme that consistently improves GPT-4o on MLVU, LongVideoBench, and
Video-MME. **BOLT** ([CVPR 2025](https://arxiv.org/abs/2503.21483)) selects frames via
inverse transform sampling on query-frame similarity, lifting Video-MME 53.8→56.1 and
MLVU 58.9→63.4 training-free.

**Takeaway:** query-aware selection beats uniform at the same or smaller frame budget;
expect +1–5 points on strong MLLMs and far more on weak ones.

## 2.2 Token compression inside the model (fewer tokens per frame)

The ancestor is BLIP-2's **Q-Former** ([ICML 2023](https://arxiv.org/abs/2301.12597)):
32 learned queries compress each image into a fixed token budget. **LLaMA-VID**
([ECCV 2024](https://arxiv.org/abs/2311.17043)) pushes this to **2 tokens per frame** (a
context token + a content token), enabling hour-long video input. **Chat-UniVi**
([CVPR 2024 Highlight](https://arxiv.org/abs/2311.08046)) uses a shared set of dynamic,
multi-scale visual tokens for both images and videos. **TokenPacker**
([2024](https://arxiv.org/abs/2407.02392), [repo](https://github.com/CircleRadon/TokenPacker))
replaces heavy resamplers with a coarse-to-fine "region-to-point" projector, compressing
visual tokens by **75%–89%** at comparable accuracy. **LongVU** (Meta,
[arXiv 2024](https://arxiv.org/abs/2410.17434)) compresses spatiotemporally: DINOv2
similarity drops redundant frames, a text-guided cross-modal query prunes per-frame
features, and temporal dependencies drive spatial token reduction — aimed squarely at
hour-long Video-MME/MLVU. **Video-XL** ([arXiv 2024](https://arxiv.org/abs/2409.14485))
assigns a Visual Summarization Token per video interval whose KV state carries that
interval's information, reporting **minimal loss at 16x compression** and thousands of
frames on a single A100.

**Takeaway:** 2–64 tokens/frame works for semantic QA; the tradeoff is fine detail (OCR,
small objects), so compress uniformly but keep full resolution for the few frames that
matter.

## 2.3 Streaming and memory for hour-long video

**MovieChat** ([CVPR 2024](https://arxiv.org/abs/2307.16449)) applies the
Atkinson–Shiffrin memory model: redundant dense tokens are consolidated into short-term
and long-term memory, and it ships MovieChat-1K (1K long videos, 14K annotations).
**Flash-VStream** ([arXiv 2024](https://arxiv.org/abs/2406.08085)) simulates human memory
with a compressed long-term buffer plus a short-term buffer, processing video streams in
real time with much lower latency and VRAM (and introduces VStream-QA). **ReKV**
([ICLR 2025](https://arxiv.org/abs/2503.00540)) is training-free: it uses sliding-window
attention, offloads KV caches to RAM/disk, and retrieves only query-relevant KV caches —
so encoding and answering can run as separate processes and GPUs, making streaming QA over
very long videos practical. Google has converged on the same shape commercially: Gemini's
agentic video mode dynamically explores the timeline and selectively inspects transcripts
rather than ingesting everything ([Gemini docs](https://ai.google.dev/gemini-api/docs/video-understanding)).

**Takeaway:** keep a constant-size memory of the past and retrieve against cached
representations instead of re-encoding the video per question.

## 2.4 Transcript-centric and hybrid approaches (the biggest lever)

The strongest quantitative evidence: on Video-MME, **"Subtitles and audios provide
greater assistance in understanding long videos compared to short videos"** — adding
subtitles improves performance by **+2.8% on short but +10.1% on long videos** (GPT-4o
long: 65.3→72.1; Gemini 1.5 Pro overall: 75.0→81.3), and "subtitles consistently provide
greater improvement than audio" ([Video-MME](https://arxiv.org/abs/2405.21075)). LLoVi
([EMNLP 2024](https://arxiv.org/abs/2312.17235)) showed an LLM reasoning over densely
generated *textual* captions alone reaches 50.3% on EgoSchema — beating the prior SOTA by
18.1% at the time — and VideoAgent's paper itself credits that "simply captioning
videos... and prompting an LLM with these captions can serve as a strong baseline"
([paper](https://arxiv.org/abs/2403.10517)). For speech-heavy video, an ASR transcript is
roughly 1–2 orders of magnitude cheaper per second than frames and directly answers "what
was said/why" questions.

**Takeaway:** for dialogue-driven video, transcript-first QA answers most queries; frames
are only needed for visually grounded ones (who/where/what-on-screen), which is exactly
where benchmarks show text alone can't help.

## 2.5 What the 2024–2025 long-video benchmarks show

**Video-MME** (900 videos / 254 hours / 2,700 QA): all MLLMs degrade as duration grows
([arXiv](https://arxiv.org/abs/2405.21075)). **MLVU** (9 tasks, 1,730 videos, ~15-min
average, 3 min–2 h): "all existing methods... exhibit severe performance degradation when
handling longer videos"; GPT-4o averages 54.5% on multi-choice and only **42.9% on Needle
QA**, and most models "fail entirely in action order and action count"
([arXiv](https://arxiv.org/abs/2406.04264)). **LVBench**
([ICCV 2025](https://arxiv.org/abs/2406.08035); videos up to 2 h, avg ~68 min) reports
humans at 94.4% vs GPT-4o 48.9% and Gemini-2.5-Pro 67.4%, with temporal grounding the
weakest capability ([repo](https://github.com/THUDM/LVBench)). The unsolved frontier is
precisely sparse-evidence retrieval and temporal/state reasoning at hour scale — not
perception of any single frame.

## Practical recipe for a transcript-first video QA CLI

1. Run ASR first; store timestamped segments. Answer text-answerable queries from
   transcript alone (cheap, and worth ~10 points on long speech-heavy videos per Video-MME).
2. Route the query: if it needs on-screen evidence (visual entities, text, motion), go to
   frames; embed transcript segments and use CLIP/SigLIP query-frame similarity for
   candidate timestamps (CLIP scoring is <1% of pipeline compute per VideoAgent).
3. Keep a uniform 1-frame-per-30–60s thumbnail strip for coverage/state, then select
   ~8–16 query-relevant frames (VideoAgent/VideoTree budget), at adaptive resolution
   (Q-Frame).
4. Encode skim layers at 2 tokens/frame (LLaMA-VID style); spend full tokens only on the
   selected frames.
5. Cache encoded features and KV states (ReKV) so follow-up questions don't re-encode.
6. If confidence is low, loop: retrieve more transcript segments or adjacent frames
   (agentic refinement).
7. Evaluate tokens-per-question alongside accuracy on Video-MME-long, MLVU (especially
   Needle QA, Action Order/Count), and LVBench-style hour-long needles.

*Section 6 applies this recipe to a real CLI project:
[06-case-study-video-qa-cli.md](06-case-study-video-qa-cli.md).*
