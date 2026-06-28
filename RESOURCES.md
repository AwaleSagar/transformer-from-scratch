# Beginner's Resource Guide

> A companion to *Building a Transformer From Scratch*. Everything here is hand-picked for **first-time learners** — verified, current as of June 2026, and mapped to the 12 blocks of this series.
>
> ✅ = link checked & live · 📺 = video · 📄 = paper/report

Each entry says *why* it helps a beginner and which block it pairs with. Skim, don't binge — come back as you hit each concept.

---

## Table of Contents

1. [Before You Start](#1-before-you-start)
2. [Per-Block Concept Deep-Dives](#2-per-block-concept-deep-dives)
3. [The 2026 Model Landscape](#3-the-2026-model-landscape)
4. [Where to Go Next (Learning Path)](#4-where-to-go-next-learning-path)
5. [Run a Real Model Locally](#5-run-a-real-model-locally)
6. [Companion Tutorials & Repos](#6-companion-tutorials--repos)
7. [Original Papers](#7-original-papers)

---

## 1. Before You Start

If you can read Python and remember a little matrix multiplication, you're ready. Warm up with these visual explanations before touching the code — they build the mental model the code will then make concrete.

- ✅ **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** — Jay Alammar
  The single best first read. Walks through the whole architecture with pictures of the actual matrices flowing through it. Read this, *then* open `01_tokenizer.py`.
- ✅ 📺 **[Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc)** — 3Blue1Brown, *Deep Learning Chapter 6*
  The famous animated explanation of attention. The visual intuition here is what makes Block 3 (`03_attention.py`) click.
- ✅ **[A Visual Guide to the Attention Mechanism](https://codecompass00.substack.com/p/visual-guide-attention-mechanism-transformers)** — Code Compass
  A slower, more recent written companion to the video above; great if you prefer reading over watching.

---

## 2. Per-Block Concept Deep-Dives

For each block, the best beginner-friendly deep-dive(s) beyond what's in the file's own comments.

### Block 1–2 — Tokenizer & Embeddings
- The Illustrated Transformer (above) covers positional encoding with diagrams. Pair the sinusoidal formulas in `02_embeddings.py` with its pictures.

### Block 3 — Scaled Dot-Product Attention
- ✅ 📺 **[Attention is all you need (Transformer)](https://www.youtube.com/watch?v=bCz4OMemCcA)** — The AI Hacker
  Walks every layer including the Q·Kᵀ → softmax → ×V pipeline. Good reinforcement for the math in Block 3.

### Block 4 — Multi-Head Attention
- (See the Illustrated Transformer's multi-head section; this repo's Block 4 comments are unusually complete.)

### Block 5 — FFN + LayerNorm
- **[Feed-Forward Networks in LLMs](https://outcomeschool.com/blog/feed-forward-networks-in-llms)** — Outcome School
  Clean explanation of why the FFN expands 4× and is the model's "knowledge store." Pairs with `05_feedforward_layernorm.py`.
- 📺 **[What Happens After Attention in Transformers?](https://www.youtube.com/watch?v=db4p0IX2RmM)** — video on the FFN's role.

### Block 6–8 — Block, Full Model, Training
- These blocks are about assembly and training mechanics. The [learning path](#4-where-to-go-next-learning-path) resources (Karpathy, Raschka) are the natural next step once they compile.

### Block 9 — Rotary Position Embeddings (RoPE)
- ✅ **[Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/)** — EleutherAI
  From the lab that stress-tests these things. Explains *why* RoPE makes attention depend on relative distance — exactly the property proven in `09_rope.py`.
- ✅ **[How Transformers Encode Position: PE & RoPE Made Simple](https://medium.com/@lepicardhugo/how-transformers-encode-position-pe-rope-made-simple-024d5e03fa03)** — Hugo Lepic
  Builds from binary → sinusoidal → RoPE, so you see the evolution this repo mirrors (Block 2's encoding → Block 9's rotation).
- ✅ **[Positional Embeddings: A Math Guide to RoPE](https://towardsdatascience.com/positional-embeddings-in-transformers-a-math-guide-to-rope-alibi/)** — Towards Data Science
  For when you want the rotation-matrix math written out line by line.

### Block 10 — RMSNorm + SwiGLU
- **[Gated Linear Units: The FFN Architecture Behind Modern LLMs](https://mbrenndoerfer.com/writing/gated-linear-units-swiglu-transformer-ffn)** — Max Brenndoerfer
  Focused, readable walkthrough of how the SwiGLU *gate* works — matches the gating code in `10_rmsnorm_swiglu.py`.
- **[Transformer Design Guide (Part 2: Modern Architecture)](https://rohitbandaru.github.io/blog/Transformer-Design-Guide-Pt2/)** — Rohit Bandaru
  Covers GLU gating and the broader modern recipe; good for seeing where RMSNorm + SwiGLU fit in the 2026 stack.

### Block 11 — GQA + KV Cache
- ✅ **[Grouped-Query Attention (GQA)](https://sebastianraschka.com/llms-from-scratch/ch04/04_gqa/)** — Sebastian Raschka
  Raschka's own chapter on GQA — the clearest "keep query heads, share K/V heads" explanation, with the memory math that Block 11 prints out.
- **[Introduction to KV Cache Optimization Using GQA](https://pyimagesearch.com/2025/10/06/introduction-to-kv-cache-optimization-using-grouped-query-attention/)** — PyImageSearch
  Shows the KV-cache-size and inference-time savings with figures — directly comparable to the MHA/GQA/MQA exercise in the README.
- **[What is grouped query attention (GQA)?](https://www.ibm.com/think/topics/grouped-query-attention)** — IBM
  Short, no-nonsense reference definition.

### Block 12 — Modern Transformer
- This block ties Blocks 9–11 together. After it, jump to the [2026 landscape](#3-the-2026-model-landscape) to see these pieces in real production models.

---

## 3. The 2026 Model Landscape

What's actually running in production today, and how each one uses the techniques from Blocks 9–12. Verified June 2026.

### The consensus stack (read this first)
- ✅ **[The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)** — Sebastian Raschka
  The survey that inspired Blocks 9–12. Compares how every major model handles positions, norms, FFN, and attention. If you read one thing, read this.

### Llama 4 (Meta) — MoE + iRoPE
- ✅ **[The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)** — Meta AI (official)
  First Llama to use Mixture-of-Experts: 128 routed experts + a shared expert, alternating dense/MoE layers. Uses **iRoPE** (interleaved RoPE, with some layers dropping position info — "NoPE").
- **[Llama 4: The Challenges of Creating a Frontier-Level LLM](https://cameronrwolfe.substack.com/p/llama-4)** — Cameron Wolfe
  Beginner-friendly deep dive on the MoE conversion.

### Qwen3 (Alibaba) — dense + MoE
- ✅ **[Understanding and Implementing Qwen3 From Scratch](https://magazine.sebastianraschka.com/p/qwen3-from-scratch)** — Sebastian Raschka
  A *from-scratch* build of a real production model — the spiritual big sibling of this repo's Block 12.
- ✅ 📄 **[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)** — 0.6B–235B params, dense and MoE variants, "thinking mode."

### DeepSeek V3 / R1 — MLA + MoE + reasoning
- ✅ 📄 **[DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)** — introduces **Multi-head Latent Attention (MLA)** (compresses the KV cache further than GQA) + DeepSeekMoE. 671B total / ~37B active.
- **[The Inner Workings of DeepSeek-V3](https://mccormickml.com/2025/02/12/the-inner-workings-of-deep-seek-v3/)** — Chris McCormick
  Excellent beginner walkthrough of MLA — read this alongside Block 11's KV-cache math to see where the field is heading.

### Gemma 3 (Google) — sliding-window attention
- ✅ 📄 **[Gemma 3 Technical Report](https://arxiv.org/html/2503.19786v1)** — 1–27B, multimodal.
- ✅ **[Gemma explained: What's new in Gemma 3](https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/)** — Google
  The **5:1 interleaving** pattern: 5 local (sliding-window) attention layers per 1 global layer — the practical answer to "long contexts get expensive" hinted at in the field guide.

### gpt-oss (OpenAI) — open weights + reasoning
- ✅ **[Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/)** — OpenAI (official)
  OpenAI's first open-weight models since GPT-2: `gpt-oss-120b` (MoE) and `gpt-oss-20b`, built for reasoning.
- **[From GPT-2 to gpt-oss: Analyzing the Architectural Advances](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the)** — Sebastian Raschka
  Traces every change from the architecture in Blocks 1–8 to a 2026 model. Highly relevant.

### Where the frontier is still moving
- **[The End of LLMs As We Know Them (2026)](https://medium.com/@aftab001x/the-end-of-llms-as-we-know-them-why-2026-marks-the-beginning-of-ais-next-architecture-revolution-902ee29484f7)** — on the shift toward hybrid/linear attention for million-token contexts. This is the one area the architecture has *not* converged (yet).

---

## 4. Where to Go Next (Learning Path)

You've built a transformer in raw NumPy — that puts you ahead of most people using these models. A sensible 2026 route, in order:

1. ✅ **[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)** — Andrej Karpathy
   Build GPT-2 in PyTorch with autograd. The natural next step after Blocks 1–8 — same ideas, now with a framework doing the gradients for you. His `nanoGPT` / `nanochat` repos are the obvious next codebase.
2. ✅ **[Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter1/1)** — free; learn the ecosystem (tokenizers, datasets, `transformers`, fine-tuning) that everyone in industry uses.
3. ✅ **[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)** — Sebastian Raschka, Manning
   Book + code; the most thorough "build it yourself" treatment in print.
4. ✅ **[The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)** — Raschka's survey (also in Section 3). Read after you understand one model, to see how they all vary.
5. **Run a real model locally** — see [Section 5](#5-run-a-real-model-locally).
6. **Then specialize** — fine-tuning (LoRA), RAG, agents, or efficiency research.

---

## 5. Run a Real Model Locally

Connect what you built (KV cache, context window, quantization) to a real model you can actually run on your laptop. Pick **one** tool:

- ✅ **[Ollama](https://ollama.com)** — the easiest start. One command: `ollama run qwen3` or `ollama run gemma3`. Best for beginners; wraps `llama.cpp` with simple model management.
  - **[How to Run LLMs Locally with Ollama](https://www.freecodecamp.org/news/run-and-customize-llms-locally-with-ollama/)** — freeCodeCamp (written walkthrough)
  - 📺 **[Learn Ollama in 15 Minutes](https://www.youtube.com/watch?v=UtSSMs6ObqY)** — quick video intro
- **LM Studio** — graphical app; great if you prefer a UI over the terminal. Same `llama.cpp` engine under the hood.
- **llama.cpp** — the raw engine Ollama/LM Studio build on. More setup, ~10–20% faster, full control. Step up to this once you're comfortable.
  - 📺 **[Ollama vs LM Studio vs llama.cpp: Which Should You Use?](https://www.youtube.com/watch?v=crXFOd7gG_I)** — comparison to help you choose.

> **Beginner tip:** start with a small model (Qwen3 4B or Gemma 3 4B). While it generates, remember that the *KV cache* (Block 11) is what makes each new token fast, and *quantization* shrinks the weights so they fit in your RAM.

---

## 6. Companion Tutorials & Repos

Other well-regarded "build a transformer from scratch" resources — useful to cross-check your understanding or see the same ideas in a different style.

- 📺 **[Coding a Transformer from scratch on PyTorch](https://www.youtube.com/watch?v=ISNdQcPhsts)** — peltarion-style, full forward + backward with PyTorch.
- **[Transformer from Scratch (in PyTorch)](https://www.mislavjuric.com/transformer-from-scratch-in-pytorch/)** — Mislav Jurić, written version of the above style.
- **[Building Transformer Models from Scratch (10-day mini-course)](https://machinelearningmastery.com/building-transformer-models-from-scratch-with-pytorch-10-day-mini-course/)** — Machine Learning Mastery, structured day-by-day.
- **[Implementing Transformer from Scratch — Step-by-Step](https://discuss.huggingface.co/t/tutorial-implementing-transformer-from-scratch-a-step-by-step-guide/132158)** — Hugging Face forum tutorial (encoder-decoder flavor).
- **[Build a Transformer from Scratch using NumPy — Part 1](https://ai.plainenglish.io/build-a-transformer-from-scratch-using-numpy-part-1-e11aac54f7e9)** — a fellow NumPy-only approach; good to compare against this repo.
- ✅ **[Understanding and Implementing Qwen3 From Scratch](https://magazine.sebastianraschka.com/p/qwen3-from-scratch)** — the "graduate" version: building a *real* 2026 model, not a toy.

> **How this repo differs:** most tutorials use PyTorch's autograd, which hides the backward pass. This repo hand-writes every gradient (with numerical checks) — the point is to *see* the math, not ship a model.

---

## 7. Original Papers

The primary sources behind every block. All verified live ✅.

- ✅ 📄 **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** — Vaswani et al., 2017 (the original; Blocks 1–8)
- ✅ 📄 **[RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)** — Su et al., 2021 (RoPE, Block 9)
- ✅ 📄 **[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)** — Shazeer, 2020 (SwiGLU, Block 10)
- ✅ 📄 **[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)** — Zhang & Sennrich, 2019 (RMSNorm, Block 10)
- ✅ 📄 **[GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)** — Ainslie et al., 2023 (Block 11)
- ✅ **[The Crystallization of Transformer Architectures (2017–2025)](https://jytan.net/blog/2025/transformer-architectures/)** — Jun Yu Tan (the 53-model survey cited in the field guide)

---

*Last verified: June 2026. Built with live web research — every ✅ link was checked at publication time.*
