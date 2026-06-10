# Transformers in 2026 — A Beginner's Field Guide

> Companion to the code series. Blocks 1–8 teach the **2017 architecture**; Blocks 9–12 upgrade it to the **2026 architecture**. This document explains what changed in between, what's running in production today, and where to go next.

---

## The Big Picture

The transformer (Vaswani et al., 2017) won. Nearly a decade later, every major AI system — ChatGPT, Claude, Gemini, Llama, Qwen, DeepSeek — is still a transformer at its core. What's remarkable is **how little** the core changed: it's still embeddings → attention + feed-forward blocks → next-token prediction, exactly what you build in this repo.

What *did* change is a set of refinements that, by ~2023, converged into a near-universal recipe. A 2025 analysis of 53 major models ([Tan, 2025](https://jytan.net/blog/2025/transformer-architectures/)) found that almost all of them settled on the same stack:

| Component | 2017 original (Blocks 1–8) | 2026 consensus (Blocks 9–12) |
|---|---|---|
| Positions | Sinusoidal encodings, **added** to embeddings | **RoPE** — rotate Q/K inside attention |
| Normalization | LayerNorm (post-norm originally) | **RMSNorm**, pre-norm (+ QK-Norm in newer models) |
| Feed-forward | ReLU/GELU, 2 matrices, biases | **SwiGLU** — gated, 3 matrices, no biases |
| Attention | Multi-Head (every head has K,V) | **GQA** — query heads share K/V heads |
| Inference | Recompute everything per token | **KV cache** — compute each token once |
| Scale-up trick | Just make it bigger | **Mixture-of-Experts** — more params, same compute |

None of these changes the *idea* of the transformer. They make it cheaper to train, cheaper to run, and stable at enormous scale.

---

## The Upgrades, Explained Simply

### 1. RoPE — Rotary Position Embeddings *(Block 9)*

Instead of adding a "position vector" to each word, RoPE **rotates** each query and key vector by an angle proportional to its position. The payoff: attention scores end up depending only on the *distance between* two tokens, not their absolute positions — which is what language actually needs ("the adjective before this noun", not "position 7"). It also extends to long documents far more gracefully, which is why every long-context trick (YaRN, etc.) builds on it. The very newest models (Llama 4, gpt-oss variants, Kimi) even drop position info entirely in some layers ("NoPE") for better length generalization.

### 2. RMSNorm *(Block 10)*

LayerNorm, but skip subtracting the mean and skip the bias term. Half the parameters, less compute, trains just as well. Recent models (OLMo 2, Gemma 3) also add **QK-Norm** — an extra RMSNorm applied to queries and keys inside attention — to keep training stable.

### 3. SwiGLU *(Block 10)*

The feed-forward network gets a **gate**: one projection decides *what* information to pass, another decides *how much* of it passes, multiplied element-wise. Smooth SiLU activation replaces ReLU. Discovered by Shazeer (2020), who famously attributed its success to "divine benevolence" — it just works better at equal parameter count.

### 4. GQA + KV Cache *(Block 11)*

When generating, a model caches the keys and values of all previous tokens so each new token costs one cheap step (the **KV cache**). At long contexts that cache becomes the main memory hog, so **Grouped-Query Attention** lets, say, 32 query heads share 8 K/V heads — a 4× smaller cache with almost no quality loss. DeepSeek pushes further with **Multi-Head Latent Attention (MLA)**, compressing K/V into a small latent vector.

### 5. Mixture-of-Experts (MoE)

The biggest 2026 models replace the single feed-forward network in each block with **many** ("experts") and route each token to only the top few. DeepSeek V3/R1: 671B total parameters, only ~37B *active* per token. You get the knowledge capacity of a huge model at the inference cost of a small one. Used by DeepSeek, Qwen3-MoE, Llama 4, gpt-oss, Kimi, MiniMax, and (reportedly) most frontier closed models.

### 6. The long-context frontier (still in flux)

Full attention costs grow quadratically with sequence length. 2025–2026 models mix in **sliding-window attention** (Gemma 3 — each token only sees a local window in most layers) and **linear/hybrid attention** (Qwen3-Next, Kimi Linear, Mamba-transformer hybrids) to reach million-token contexts. This is the one area where the architecture has *not* converged yet — worth watching.

---

## What Else Changed (Beyond Architecture)

The architecture is maybe a third of the modern story. The rest:

**Tokenizers** — real models use Byte-Pair Encoding (BPE) over bytes, with vocabularies of ~32K–256K subword pieces, not the word-level tokenizer of Block 1.

**Training pipeline** — modern assistants are built in stages: *pretraining* (next-token prediction on trillions of tokens — what Block 8 does in miniature), then *post-training*: supervised fine-tuning on instruction data, and reinforcement learning (RLHF and, increasingly, RL on verifiable tasks like math and code). The optimizer is AdamW with learning-rate warmup + cosine decay, not plain SGD.

**Reasoning models** — the headline shift of 2025–2026. Models like DeepSeek-R1, OpenAI's o-series, Claude's extended thinking, and Qwen3's "thinking mode" are trained to generate long internal chains of thought before answering. Architecture: same transformer. Difference: post-training and inference-time compute.

**Multimodality & agents** — frontier models now handle images, audio, and video, and are trained to use tools (search, code execution, file editing) in a loop. Again: same transformer underneath.

---

## The 2026 Model Landscape (beginner's map)

**Closed/frontier (API access):** OpenAI GPT-5 series, Anthropic Claude (Opus/Sonnet), Google Gemini 3. You can't see their weights, but public information suggests the same family of techniques.

**Open-weight (you can download and run):**

- **Llama (Meta)** — the model that mainstreamed open weights; Llama 4 is MoE
- **Qwen3 (Alibaba)** — dense + MoE variants, very strong, hugely popular for fine-tuning
- **DeepSeek V3/R1** — MLA + MoE, the model that proved open reasoning models work
- **Gemma 3 (Google)** — efficient sliding-window attention, great small sizes
- **gpt-oss (OpenAI)**, **Mistral**, **OLMo 2** (fully open, including training data — great for learning)

For a beginner, small open models (1–8B) are gold: you can run them locally (via `llama.cpp`, Ollama, or LM Studio), inspect them, and fine-tune them on a single GPU.

---

## Your Learning Path From Here

You've built a transformer from raw NumPy — that puts you ahead of most people using these models. A sensible 2026 route:

1. **Finish this repo** — Blocks 1–8, then 9–12. Do the exercises.
2. **Karpathy's "Zero to Hero"** videos — build GPT-2 in PyTorch with autograd ([karpathy.ai/zero-to-hero](https://karpathy.ai/zero-to-hero.html)); his `nanoGPT`/`nanochat` repos are the natural next codebase after this one.
3. **Hugging Face LLM Course** — free; learn the ecosystem everyone uses ([huggingface.co/learn](https://huggingface.co/learn/llm-course/en/chapter1/1)).
4. **Sebastian Raschka's *Build a Large Language Model (From Scratch)*** — book + code; his ["Big LLM Architecture Comparison"](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) is the best single survey of everything in this guide.
5. **Run a real model locally** — install Ollama, pull a small Qwen3 or Gemma 3, and connect what you see (KV cache size, context window, quantization) to Blocks 9–12.
6. **Then specialize** — fine-tuning (LoRA), RAG, agents, or training/efficiency research.

---

## Exercises (2026 edition)

1. **Add QK-Norm** to Block 11: apply an RMSNorm to Q and K (per head) before RoPE. Compare attention score ranges with and without.
2. **Minimal MoE**: replace the SwiGLU in Block 12 with 4 SwiGLU "experts" and a learned router (`softmax(x @ W_router)`, pick top-1). Verify only one expert runs per token.
3. **Sliding-window attention**: in Block 11, mask attention to the previous 4 tokens only. How does the KV cache requirement change?
4. **Measure the cache**: print `cache_k.nbytes` in Block 11 as you generate 50 tokens. Plot memory vs. sequence length for MHA / GQA / MQA configs.
5. **Tie it together**: port Blocks 9–12 to PyTorch with autograd and train on the Block 1 corpus. Does the modern stack converge faster than the 2017 one?

---

## Sources & Further Reading

- Vaswani et al., *Attention Is All You Need* (2017) — [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Su et al., *RoFormer: Rotary Position Embedding* (2021) — [arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
- Shazeer, *GLU Variants Improve Transformer* (2020) — [arxiv.org/abs/2002.05202](https://arxiv.org/abs/2002.05202)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer* (2023) — [arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019) — [arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)
- Raschka, *The Big LLM Architecture Comparison* (2025) — [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
- Tan, *The Crystallization of Transformer Architectures 2017–2025* — [jytan.net](https://jytan.net/blog/2025/transformer-architectures/)
- Hugging Face LLM Course — [huggingface.co/learn](https://huggingface.co/learn/llm-course/en/chapter1/1)

*Last updated: June 2026, researched via live web sources.*
