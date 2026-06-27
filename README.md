# Building a Transformer From Scratch

![Python](https://img.shields.io/badge/python-3.8+-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/dependencies-numpy%20only-013243?logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Parameters](https://img.shields.io/badge/parameters-~14K-orange)
![Trains in](https://img.shields.io/badge/trains%20in-~3s-success)
![Blocks](https://img.shields.io/badge/blocks-12-purple)

> **A 12-part educational series** — build a GPT-style language model block by block in Python using only NumPy, then upgrade it to the modern 2026 architecture (RoPE, RMSNorm, SwiGLU, GQA, KV cache). Every forward *and* backward pass is hand-written so you can see exactly how gradients flow.

**Loss drops from 2.65 → 0.18 in ~3 seconds on a CPU** — every weight matrix is small enough to print.

```text
  Epoch   1/100  Loss: 2.6523   "the cat chased the big and the cat..."
  Epoch  100/100  Loss: 0.1808   "the cat watched the bird and the dog..."
                          ▼  93.2% reduction
```

<p align="center">
  <a href="#quick-start"><strong>Quick Start</strong></a> ·
  <a href="#architecture-overview">Architecture</a> ·
  <a href="#series-roadmap">Roadmap</a> ·
  <a href="#block-by-block-deep-dive">Deep Dive</a> ·
  <a href="TRANSFORMERS_2026.md">2026 Field Guide</a>
</p>

---

## Table of Contents

1. [Why This Series?](#why-this-series)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Series Roadmap](#series-roadmap)
5. [Block-by-Block Deep Dive](#block-by-block-deep-dive)
6. [What to Expect](#what-to-expect)
7. [Model Configuration](#model-configuration)
8. [Exercises](#exercises-for-students)
9. [Key Design Decisions](#key-design-decisions)
10. [File Structure](#file-structure)
11. [Contributing](#contributing)
12. [License](#license)
13. [References](#references)

---

## Why This Series?

Most transformer tutorials use PyTorch or TensorFlow, which hide the internals behind autograd and pre-built layers. This series strips everything down to raw matrix operations so you can:

- ✅ **See** what `softmax`, `attention`, and `LayerNorm` actually compute
- ✅ **Trace** how gradients propagate backward through every layer
- ✅ **Verify** each backward pass with a numerical gradient check
- ✅ **Run** the entire model on a CPU in under 5 seconds
- ✅ **Inspect** every weight matrix (they're small enough to print)
- ✅ **Understand** how 2017 became 2026 (RoPE, RMSNorm, SwiGLU, GQA)

The model is deliberately tiny (~14,200 parameters, 24-word vocabulary) — it's a teaching tool, not a production system.

*(back to [top](#building-a-transformer-from-scratch))*

---

## Quick Start

### Prerequisites

- Python 3.8+
- NumPy

### Install

```bash
git clone https://github.com/AwaleSagar/transformer.git
cd transformer
pip install -r requirements.txt      # just numpy
```

### Run

Each block is self-contained with a runnable self-test and gradient checks:

```bash
python3 01_tokenizer.py              # vocabulary, encoding, training pairs
python3 03_attention.py              # attention weights + gradient check
python3 07_transformer.py            # full model forward/backward pass
python3 08_train_and_generate.py     # train + generate text  ← the fun one
```

Or, in one line, train the whole model:

```bash
python3 08_train_and_generate.py
```

Expected output (~3 seconds on a modern CPU):

```
  Epoch    1/100  Loss: 2.6523  Sample: "the cat chased the big and..."
  Epoch   50/100  Loss: 0.1980  Sample: "the cat the cat watched the bird..."
  Epoch  100/100  Loss: 0.1808  Sample: "the cat watched the bird and the dog..."

  Loss: 2.65 → 0.18 (93.2% reduction)
```

### Run the modern (2026) blocks

```bash
python3 09_rope.py                # RoPE: relative-position property proof
python3 10_rmsnorm_swiglu.py      # RMSNorm vs LayerNorm, SwiGLU gating
python3 11_gqa_kv_cache.py        # GQA + KV cache, memory math
python3 12_modern_transformer.py  # full Llama-style block, cached generation
```

*(back to [top](#building-a-transformer-from-scratch))*

---

## Architecture Overview

```text
Token IDs  [2, 3, 4, 5, 2, 6, ...]
    │
    ▼
┌──────────────────────────────┐
│  Token Embedding (24 × 32)   │   Block 2
│  + Positional Encoding       │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  ┌────────────────────────┐  │
│  │ LayerNorm              │  │
│  │ Multi-Head Attention   │  │   Blocks 3–6
│  │ + Residual Connection  │  │
│  ├────────────────────────┤  │
│  │ LayerNorm              │  │
│  │ Feed-Forward Network   │  │
│  │ + Residual Connection  │  │
│  └────────────────────────┘  │
│         × 1 layer            │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Final LayerNorm             │   Block 7
│  Linear → Logits (32 → 24)   │
└──────────────────────────────┘
    │
    ▼
  Softmax → next-word probabilities
```

**Decoder-only** (GPT-style) — uses causal masking so each token can only attend to itself and earlier tokens.

*(back to [top](#building-a-transformer-from-scratch))*

---

## Series Roadmap

Each file introduces one concept, imports from the previous files, and includes a runnable self-test.

| # | File | Concept | What You'll Build |
|---|------|---------|-------------------|
| 1 | `01_tokenizer.py` | **Vocabulary & Tokenization** | `Tokenizer` class — `fit()`, `encode()`, `decode()`, training data generation |
| 2 | `02_embeddings.py` | **Embeddings** | Token lookup table + sinusoidal positional encoding |
| 3 | `03_attention.py` | **Scaled Dot-Product Attention** | `softmax()`, causal mask, `attention_forward()`, `attention_backward()` |
| 4 | `04_multi_head_attention.py` | **Multi-Head Attention** | Split/concat heads, Q/K/V projections, output projection |
| 5 | `05_feedforward_layernorm.py` | **FFN + Layer Normalization** | `Linear`, `ReLU`, `FeedForward`, `LayerNorm` — each with forward & backward |
| 6 | `06_transformer_block.py` | **Transformer Block** | Pre-norm architecture, residual connections |
| 7 | `07_transformer.py` | **Full Transformer** | Stack all blocks, cross-entropy loss, full forward/backward |
| 8 | `08_train_and_generate.py` | **Training & Generation** | SGD optimizer, training loop, greedy & temperature-based text generation |

### Modern series (2026) — forward-pass focused

| # | File | Concept | What You'll Build |
|---|------|---------|-------------------|
| 9 | `09_rope.py` | **Rotary Position Embeddings** | RoPE rotation, proof that scores depend only on relative distance |
| 10 | `10_rmsnorm_swiglu.py` | **RMSNorm + SwiGLU** | The modern norm and gated FFN that replaced LayerNorm/ReLU |
| 11 | `11_gqa_kv_cache.py` | **GQA + KV Cache** | Grouped-Query Attention with cached generation, verified against the full pass |
| 12 | `12_modern_transformer.py` | **Modern Transformer** | A Llama-style mini-model: RoPE + RMSNorm + SwiGLU + GQA + tied embeddings |

**Dependency chain:** `01 → 02 → 03 → 04 → 05 → 06 → 07 → 08`, then `09 → 10 → 11 → 12` (uses `03` and `09`).

See [`TRANSFORMERS_2026.md`](TRANSFORMERS_2026.md) for the full story of what changed since 2017 — and what we deliberately left out (MoE, sliding-window attention, QK-Norm) as exercises.

*(back to [top](#building-a-transformer-from-scratch))*

---

## Block-by-Block Deep Dive

### Block 1 — Tokenizer

**Concept:** Convert text to numbers and back.

A neural network can't process raw text — it needs integers. The tokenizer builds a word→ID mapping:

```
"the cat sat" → [2, 3, 4]
[2, 3, 4] → "the cat sat"
```

We also prepare training data: sliding windows where the input is `words[i:i+16]` and the target is `words[i+1:i+17]` (shifted by one position for next-word prediction).

**Special tokens:**
- `<PAD>` (ID=0) — padding for sequences shorter than `seq_len`
- `<UNK>` (ID=1) — placeholder for unknown words not in the vocabulary

---

### Block 2 — Token + Positional Embeddings

**Concept:** Turn integer IDs into dense vectors the model can learn from.

**Token Embedding** — A lookup table of shape `(vocab_size, d_model)`. Each word gets a 32-dimensional vector initialized with Xavier initialization. These vectors are learnable — they'll be updated during training.

**Positional Encoding** — Sine/cosine waves that encode each position uniquely:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

The final embedding is the **sum** of token embedding + positional encoding:

```
embedding("the" at position 0) ≠ embedding("the" at position 4)
```

---

### Block 3 — Scaled Dot-Product Attention

**Concept:** Let each token decide how much to "look at" every other token.

The core formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Step by step:**

1. **Scores** = Q × Kᵀ — how much each query matches each key
2. **Scale** by √d_k — prevents softmax from becoming too "peaky"
3. **Mask** — set future positions to −∞ (causal/autoregressive)
4. **Softmax** — convert to probabilities (each row sums to 1)
5. **Output** = weights × V — weighted combination of values

**Causal masking** ensures token 3 can only attend to tokens 0, 1, 2, 3 — not token 4 or beyond. This is what makes it autoregressive.

Includes a **numerical gradient check** to verify the backward pass is correct.

---

### Block 4 — Multi-Head Attention

**Concept:** Run several attention computations in parallel, each learning a different "perspective."

Instead of one big attention, we:

1. **Project** input through W_Q, W_K, W_V (learned matrices)
2. **Split** into `n_heads` smaller chunks (32 dims → 2 × 16 dims)
3. **Attend** independently per head
4. **Concatenate** results
5. **Project** through W_O

Each head can specialize — one might learn to focus on the subject, another on the verb. With 2 heads and d_model=32, each head works with 16-dimensional Q, K, V.

---

### Block 5 — Feed-Forward Network + Layer Normalization

**Concept:** Process each token's representation through a small neural network, and keep activations stable with normalization.

**Feed-Forward Network** — A two-layer MLP applied independently to each token:

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

The hidden dimension expands 4× (32 → 128) then compresses back (128 → 32). This is where the model does most of its per-token "thinking."

**Layer Normalization** — For each token, normalize across the 32 features:

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Before LayerNorm: mean varies wildly per token, std is inconsistent.
After LayerNorm: mean ≈ 0, std ≈ 1 for every token. This stability is critical for training.

---

### Block 6 — Transformer Decoder Block

**Concept:** Combine attention + FFN with residual connections and pre-normalization.

```text
Input X
  │
  ├───────────────┐
  │               │
  ▼               │
LayerNorm         │      Pre-Norm: normalize BEFORE the sublayer
  │               │
  ▼               │
Multi-Head Attn   │
  │               │
  ▼               │
+ ◄───────────────┘      Residual: output = input + sublayer(norm(input))
  │
  ├───────────────┐
  │               │
  ▼               │
LayerNorm         │
  │               │
  ▼               │
Feed-Forward      │
  │               │
  ▼               │
+ ◄───────────────┘      Residual again
  │
  ▼
Output
```

**Residual connections** prevent vanishing gradients — the gradient has a direct highway from output back to input. **Pre-Norm** (normalize before the sublayer, not after) is more stable and used by GPT-2, GPT-3, and LLaMA.

---

### Block 7 — Full Transformer Model

**Concept:** Stack everything into a complete language model.

```
Embedding → TransformerBlock × 1 → LayerNorm → Linear(32→24) → Logits
```

Also implements **cross-entropy loss** — the training signal:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(\text{correct word}_t)$$

The gradient of softmax + cross-entropy simplifies beautifully:

$$\frac{\partial \mathcal{L}}{\partial z_i} = P(i) - \mathbb{1}[i = \text{target}]$$

*(Probability minus 1 for the correct class, probability minus 0 for incorrect classes.)*

---

### Block 8 — Training & Text Generation

**Concept:** Make the model learn, then use it to write text.

**Training (SGD):**
```
for each epoch:
    for each (input, target) pair:
        logits = model.forward(input)
        loss = cross_entropy(logits, target)
        model.backward(loss_gradient)
        for param in model.parameters:
            param -= learning_rate × gradient
```

**Text Generation (Autoregressive):**
```
prompt = "the cat"
for i in range(max_tokens):
    logits = model.forward(prompt_ids)
    next_word = sample(softmax(logits[-1] / temperature))
    prompt_ids.append(next_word)
```

**Temperature** controls creativity:
- `T → 0` — always pick the most likely word (greedy, deterministic)
- `T = 1.0` — sample from the model's distribution (balanced)
- `T > 1.0` — flatter distribution, more surprising choices

*(back to [top](#building-a-transformer-from-scratch))*

---

## What to Expect

After 100 epochs of training:

| Metric | Value |
|--------|-------|
| Starting loss | ~2.65 (near random: ln(24) ≈ 3.18) |
| Final loss | ~0.18 |
| Reduction | 93.2% |
| Training time | ~3 seconds |

The model memorizes the small corpus and generates plausible sentences:

```
"the cat" → "the cat sat on the mat and the dog sat on the rug"
"a bird"  → "a bird sat on the fence and watched the cat"
"the big" → "the big dog chased a small bird around the house"
```

This is expected — a 14K-parameter model on a 157-token corpus *should* memorize. The learning objective here is the architecture, not the capacity.

*(back to [top](#building-a-transformer-from-scratch))*

---

## Model Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| `vocab_size` | 24 | Small corpus → tiny vocabulary |
| `d_model` | 32 | Small enough to print full matrices |
| `n_heads` | 2 | Minimum to demonstrate multi-head concept |
| `d_k` | 16 | `d_model / n_heads` |
| `d_ff` | 128 | Standard 4× expansion ratio |
| `n_layers` | 1 | One block is sufficient to learn the architecture |
| `seq_len` | 16 | Short context window |
| **Total params** | **~14,200** | Trains in seconds on CPU |

*(back to [top](#building-a-transformer-from-scratch))*

---

## Exercises for Students

<details>
<summary><strong>2017 edition (click to expand)</strong></summary>

1. **Increase model size** — Try `d_model=64, n_heads=4, n_layers=2`. How does training time and loss change?
2. **Write a bigger corpus** — Edit `CORPUS` in `01_tokenizer.py`. Add more sentences with new words. Does the model still converge?
3. **Implement dropout** — Add a `Dropout` layer after attention and FFN. Does it help when the corpus is larger?
4. **Add learning rate scheduling** — Start with `lr=0.1` and decay it. Does training become more stable?
5. **Visualize attention** — In Block 4, print the attention weights for a trained model. Which head focuses on which relationship?
6. **Compare Pre-Norm vs Post-Norm** — Move LayerNorm to *after* attention/FFN in Block 6. Does training become less stable?
7. **Implement beam search** — Instead of greedy/sampling, keep the top-k partial sequences at each step.
8. **Gradient analysis** — After training, inspect gradient magnitudes through the layers. Do residual connections prevent vanishing gradients?
</details>

<details>
<summary><strong>2026 edition (click to expand)</strong></summary>

1. **Add QK-Norm** to Block 11: apply an RMSNorm to Q and K (per head) before RoPE. Compare attention score ranges with and without.
2. **Minimal MoE**: replace the SwiGLU in Block 12 with 4 SwiGLU "experts" and a learned router (`softmax(x @ W_router)`, pick top-1). Verify only one expert runs per token.
3. **Sliding-window attention**: in Block 11, mask attention to the previous 4 tokens only. How does the KV cache requirement change?
4. **Measure the cache**: print `cache_k.nbytes` in Block 11 as you generate 50 tokens. Plot memory vs. sequence length for MHA / GQA / MQA configs.
5. **Tie it together**: port Blocks 9–12 to PyTorch with autograd and train on the Block 1 corpus. Does the modern stack converge faster than the 2017 one?
</details>

*(back to [top](#building-a-transformer-from-scratch))*

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **NumPy only** | Remove framework magic — students see raw matrix operations |
| **Manual backward passes** | No autograd — students trace gradient flow explicitly |
| **Pre-Norm** | More stable than Post-Norm without learning rate warmup |
| **Decoder-only** | Simpler than encoder-decoder; matches modern LLM architecture (GPT/LLaMA) |
| **Xavier init** | Simple, well-understood weight initialization |
| **No dropout** | Reduces code complexity; overfitting the tiny corpus is the goal |
| **Plain SGD** | Simplest optimizer — no momentum, Adam, or schedulers to explain |
| **Word-level tokens** | Simpler than BPE/SentencePiece; sufficient for a 24-word vocabulary |

*(back to [top](#building-a-transformer-from-scratch))*

---

## File Structure

```text
transformer/
├── README.md                      ← You are here
├── TRANSFORMERS_2026.md           ← Field guide: 2017 → 2026, models, learning path
├── requirements.txt               ← Python dependencies (numpy)
├── load_module.py                 ← Import helper (Python can't import 01_*.py directly)
├── 01_tokenizer.py                ← Block 1: Tokenizer
├── 02_embeddings.py               ← Block 2: Token + Positional Embeddings
├── 03_attention.py                ← Block 3: Scaled Dot-Product Attention
├── 04_multi_head_attention.py     ← Block 4: Multi-Head Attention
├── 05_feedforward_layernorm.py    ← Block 5: FFN + Layer Normalization
├── 06_transformer_block.py        ← Block 6: Transformer Decoder Block
├── 07_transformer.py              ← Block 7: Full Transformer + Loss
├── 08_train_and_generate.py       ← Block 8: Training Loop + Generation
├── 09_rope.py                     ← Block 9: Rotary Position Embeddings (2026)
├── 10_rmsnorm_swiglu.py           ← Block 10: RMSNorm + SwiGLU (2026)
├── 11_gqa_kv_cache.py             ← Block 11: GQA + KV Cache (2026)
└── 12_modern_transformer.py       ← Block 12: Modern Transformer Block (2026)
```

*(back to [top](#building-a-transformer-from-scratch))*

---

## Contributing

This is primarily an educational resource, but improvements are welcome — especially:

- **Bug fixes** in the math or gradient checks
- **New exercises** that teach a concept well
- **Clarifications** in the comments or docs

1. Fork the repo
2. Create a branch (`git checkout -b fix/awesome-fix`)
3. Commit (`git commit -m 'Fix gradient check in Block 3'`)
4. Push (`git push origin fix/awesome-fix`)
5. Open a Pull Request

Please verify any math change still passes the numerical gradient checks (`python3 03_attention.py`, etc.).

*(back to [top](#building-a-transformer-from-scratch))*

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

© 2026 AwaleSagar

*(back to [top](#building-a-transformer-from-scratch))*

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017 (the original transformer paper)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar's visual guide
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Pre-Norm decoder-only architecture
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Pre-Norm vs Post-Norm analysis
- [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021 (RoPE, Block 9)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020 (SwiGLU, Block 10)
- [GQA: Generalized Multi-Query Attention](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023 (Block 11)
- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) — Raschka, 2025 (survey behind Blocks 9–12)

*(back to [top](#building-a-transformer-from-scratch))*

---

*Built for education. ~14,200 parameters. Trains in 3 seconds. Every line is meant to be read.*
