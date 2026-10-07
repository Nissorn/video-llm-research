# 1. How frontier models ingest video (and what it costs in tokens)

*Researched 2026-10-07 against official API docs, pricing pages, model cards and papers.
Related: [§2](02-efficient-video-understanding.md) (making it cheaper).*

**The common pipeline.** Every system below shares one skeleton: decode the video → sample
frames at a fixed or dynamic cadence → feed each frame to a vision encoder that chops it
into patches and merges them into visual tokens → tokenize the audio track separately with
an audio encoder. So **tokens per minute = (frames/min × tokens/frame) + (60 × audio tokens/sec)**.
Providers differ mainly in who does the sampling and how aggressively frames are compressed.

| Provider | How video enters the model | Tokens per minute (arithmetic) | Max length | Primary source |
|---|---|---|---|---|
| **Google Gemini (API)** | Native. Server decodes; samples at **1 FPS**; frames and audio tokenized separately; per-request `mediaResolution` knob; YouTube URLs accepted directly | Gemini 3 table: **70 tok/frame** (default/low/medium) or **280** (high) + **25 tok/s audio** → (1×70+25) ≈ **95 tok/s ≈ 5,700 tok/min**; high: 280+25 ≈ **305 tok/s ≈ 18,300 tok/min**. Earlier-model numbers on the video page: 66/258 tok per frame, 32 tok/s audio → docs round to **~100 / ~300 tok/s** | **1M context: up to 3 h at default (low) resolution, ~1 h at high**; YouTube URLs: free tier 8 h/day, paid unlimited, ≤10 videos/request | [Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding), [Media resolution](https://ai.google.dev/gemini-api/docs/media-resolution), [Pricing](https://ai.google.dev/gemini-api/docs/pricing) |
| **OpenAI (GPT-4o/4.1/5/5.1/o3, o4-mini)** | **No native video.** You sample frames and send them as images (up to **1,500 images/request**, 512 MB payload) | Tile models: **gpt-4o/4.1: 85 base + 170/512² tile**; **gpt-5/5.1: 70 + 140**; **o3: 75 + 150**; low detail = base only. 720p @1 fps on gpt-4o high: 85+6×170 = **1,105 tok/frame → 66,300 tok/min**; low: 85×60 = **5,100 tok/min**. Patch models (o4-mini ×1.72, gpt-4.1-mini ×1.62): tokens = ceil(patches × multiplier) after shrinking to the patch budget | Bounded only by context window and the 1,500-image/request cap | [Images and vision](https://developers.openai.com/api/docs/guides/images-vision) |
| **Anthropic Claude** | **No video.** Images only (JPEG/PNG/GIF/WebP) — you extract frames yourself | Legacy (2024 docs): **tokens = (width × height)/750**, e.g. 1092² ≈ **1,590 tok** → 1 fps ≈ **95,400 tok/min**. Current docs: **⌈w/28⌉ × ⌈h/28⌉**, capped at **1,568 tok** (standard tier, 1568 px long edge) or **4,784 tok** (high-res tier, Claude 4.7+, 2576 px) | 100 images/request (200k-context models), 10 MB/image | [Vision docs](https://platform.claude.com/docs/en/docs/build-with-claude/vision); legacy formula verified via [archived Sept 2024 capture](https://web.archive.org/web/20241001000000/https://docs.anthropic.com/en/docs/build-with-claude/vision) |
| **Qwen2.5-VL (open weights)** | Native. **Dynamic FPS sampling** (default **2 fps**, 4–768 frames) with **MRoPE aligned to absolute time**; 14 px patches + **2×2 spatial merge** (28 px/token) | Per-frame video budget **128–768 tokens** → 2 fps × 128–768 = **256–1,536 tok/s (≈15k–92k tok/min)**; total video capped at ~0.9 × 128k context ≈ **115k tokens** | **">1 hour"** videos (model card); "videos of extended durations (up to hours) with **second-level event localization**" (paper abstract) | [HF model card](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct), [arXiv:2502.13923](https://arxiv.org/abs/2502.13923), [qwen-vl-utils source](https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py) |
| **Qwen3-VL (open weights)** | Native. Default **fps = 2**; 16 px patches → **32× spatial + 2× temporal compression**; Interleaved-MRoPE, text–timestamp alignment | Per second of video ≈ (H/32)(W/32): a 1024×768 frame stream → **768 tok/s**; whole-video visual-token budget **256–16,384** | **Native 256K context, expandable to 1M**; "handles books and **hours-long video** with full recall and second-level indexing" | [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL) |
| **YouTube "Ask" (product)** | Not an API pipeline: you ask in the player; Google-side systems answer about the video | Not exposed | Select videos; Android/iOS/web/TV | [Support page](https://support.google.com/youtube/answer/14110396), [YouTube blog](https://blog.youtube/inside-youtube/2024-in-youtube-ai) |

## Gemini in detail

Frames are extracted at 1 FPS and audio at 1 Kbps mono; [the video
page](https://ai.google.dev/gemini-api/docs/video-understanding) states ~100 tokens/sec of
video at default (low) resolution and ~300 at high, with metadata included. The
[media-resolution page](https://ai.google.dev/gemini-api/docs/media-resolution) documents
the Gemini 3 table (video frames 70/280, audio fixed at 25 tok/s, images 280–2,240).
**Audio and video are tokenized as separate modalities and billed per token**: for Gemini
2.5 Flash, [pricing](https://ai.google.dev/gemini-api/docs/pricing) is $0.30/1M for
text/image/video input but **$1.00/1M for audio** (2.5 Flash-Lite: $0.10 vs $0.30), i.e.
one minute at low resolution ≈ 6,000 video tokens ($0.0018) + 1,920 audio tokens ($0.0019)
≈ **$0.004/min**; Gemini 3.8 Flash charges a flat $0.75/1M for all input modalities (promo
through Dec 31, 2026). Custom frame rates (e.g. `fps: 0.5`) and clipping intervals are
supported.

## OpenAI in detail

There is no `video` content type in the current
[vision guide](https://developers.openai.com/api/docs/guides/images-vision); video
understanding means client-side frame sampling into image inputs. High detail resizes to
fit 2048², clamps the short side to 768 px, then bills base + per-tile tokens; newer
patch-based models use 32×32 px patches with per-model multipliers and patch budgets
(2,500–6,144 patches). Sampling cadence dominates cost: 1 fps of 720p on gpt-4o high
detail is ~66k tokens/min, but 1 frame every 4 s is ~16.6k, and low-detail frames cut it
to ~5k.

## Claude in detail

No video input; the practical pattern is frame extraction into ~100 images per request.
Note the [current docs](https://platform.claude.com/docs/en/docs/build-with-claude/vision)
have replaced the classic `(width × height)/750` formula with the 28×28-patch formula
above — the legacy formula was verified only against an
[archived 2024 capture](https://web.archive.org/web/20241001000000/https://docs.anthropic.com/en/docs/build-with-claude/vision).

## Open-weight Qwen in detail

Qwen2.5-VL's
[qwen-vl-utils](https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)
sets `FPS = 2.0`, `FPS_MAX_FRAMES = 768`, per-frame video tokens 128–768, and a total
budget of `0.9 × context × 28²` pixels. [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
doubles compression via 2× temporal merging, making tokens/second ≈ pixels/1024.

## YouTube "Ask"

Google's [support page](https://support.google.com/youtube/answer/14110396) says the
feature answers "questions about the video you're watching" and suggests related content
"all without leaving the video," is "built with Gemini large language models (LLMs)" that
"draw on info from YouTube and the web," and runs on Android/iOS/web/TV for select videos;
the [YouTube blog](https://blog.youtube/inside-youtube/2024-in-youtube-ai) adds quiz-style
academic questions for Premium members. Whether Google indexes transcript *and* visuals
server-side for this feature is **not publicly documented** — flagged as unverifiable from
primary sources. (Google's own Gemini API does ingest YouTube URLs server-side — free tier
capped at 8 h/day — per the
[video-understanding docs](https://ai.google.dev/gemini-api/docs/video-understanding).)

## Unverified/flagged

1. 2M-context video length limits no longer appear on the current Gemini video page (only
   1M figures are given); older 6 h/2 h figures could not be re-verified.
2. Gemini's two doc pages show slightly different frame/audio token values (66/258 + 32
   tok/s vs 70/280 + 25 tok/s) — consistent with 2.x-era vs Gemini 3 tables, both rounding
   to ~100/~300 tok/s.
3. The Qwen2.5-VL GitHub README now redirects to the Qwen3-VL repo; Qwen2.5-VL numbers
   were sourced from its HF model card, arXiv paper, and utility source code.
