"""
============================================================
Block 12: The Modern Transformer Block (2026 edition)
============================================================
Assemble Blocks 9-11 into a Llama-style decoder block and
compare it side by side with the 2017-style block from
Block 6. This is the architecture (give or take details)
behind Llama 3, Qwen3, Gemma 3, Mistral, DeepSeek, gpt-oss
and essentially every open model you can download in 2026.

  2017 block (Block 6)         2026 block (this file)
  --------------------         ----------------------
  LayerNorm                    RMSNorm
  Multi-Head Attention         Grouped-Query Attention
  + sinusoidal pos (added)     + RoPE (rotated inside attn)
  + residual                   + residual
  LayerNorm                    RMSNorm
  FFN (ReLU, biases)           SwiGLU FFN (no biases)
  + residual                   + residual

Still missing from our toy (used by the biggest models):
  - Mixture-of-Experts: replace ONE SwiGLU with MANY and
    route each token to its top-k experts → huge capacity,
    small per-token compute (DeepSeek V3: 671B params,
    only 37B active per token). Built next in Block 13.
  - Sliding-window / linear attention layers for long context
  - QK-Norm (RMSNorm on queries/keys) for training stability

The point of Blocks 1-12: a frontier model is THIS, scaled
up ~a-million-fold and trained on trillions of tokens.
============================================================
"""

import numpy as np
from load_module import load

_b10 = load("10_rmsnorm_swiglu.py")
RMSNorm, SwiGLU = _b10.RMSNorm, _b10.SwiGLU

_b11 = load("11_gqa_kv_cache.py")
GroupedQueryAttention = _b11.GroupedQueryAttention

_b3 = load("03_attention.py")
softmax = _b3.softmax


class ModernTransformerBlock:
    """Pre-norm decoder block: x + Attn(RMSNorm(x)), x + FFN(RMSNorm(x))."""

    def __init__(self, d_model, n_q_heads, n_kv_heads, d_ff, max_seq_len=64, seed=0):
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_q_heads, n_kv_heads,
                                          max_seq_len, seed=seed)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, seed=seed + 1)

    def forward(self, x, use_cache=False):
        x = x + self.attn.forward(self.attn_norm.forward(x), use_cache=use_cache)
        x = x + self.ffn.forward(self.ffn_norm.forward(x))
        return x


class ModernTransformer:
    """
    A tiny Llama-style language model (forward pass only):

      token embedding (NO positional encoding added — RoPE
      handles positions inside attention)
        → ModernTransformerBlock × n_layers
        → final RMSNorm
        → output projection (tied to the embedding, another
          common modern trick: reuse the embedding matrix)
    """

    def __init__(self, vocab_size, d_model=32, n_q_heads=4, n_kv_heads=2,
                 d_ff=None, n_layers=2, max_seq_len=64, seed=0):
        d_ff = d_ff or int(8 * d_model / 3)
        rng = np.random.default_rng(seed)
        self.embedding = rng.normal(0, 0.02, (vocab_size, d_model))
        self.blocks = [ModernTransformerBlock(d_model, n_q_heads, n_kv_heads,
                                              d_ff, max_seq_len, seed=seed + i)
                       for i in range(n_layers)]
        self.final_norm = RMSNorm(d_model)

    def forward(self, token_ids, use_cache=False):
        """token_ids: list/array of ints → logits (seq_len, vocab_size)"""
        x = self.embedding[np.asarray(token_ids)]   # no positions added!
        for block in self.blocks:
            x = block.forward(x, use_cache=use_cache)
        x = self.final_norm.forward(x)
        return x @ self.embedding.T                  # weight tying

    def reset_cache(self):
        for block in self.blocks:
            block.attn.reset_cache()

    def generate(self, prompt_ids, n_tokens, temperature=1.0, seed=0):
        """KV-cached autoregressive generation (the modern loop)."""
        rng = np.random.default_rng(seed)
        self.reset_cache()
        # Prefill: process the whole prompt once, filling the cache
        logits = self.forward(list(prompt_ids), use_cache=True)
        ids = list(prompt_ids)
        for _ in range(n_tokens):
            probs = softmax(logits[-1] / temperature)
            next_id = int(rng.choice(len(probs), p=probs))
            ids.append(next_id)
            # Decode step: ONLY the new token goes through the model
            logits = self.forward([next_id], use_cache=True)
        return ids

    def num_parameters(self):
        n = self.embedding.size
        for b in self.blocks:
            a = b.attn
            n += a.W_Q.size + a.W_K.size + a.W_V.size + a.W_O.size
            n += b.ffn.W_gate.size + b.ffn.W_up.size + b.ffn.W_down.size
            n += b.attn_norm.gamma.size + b.ffn_norm.gamma.size
        n += self.final_norm.gamma.size
        return n


# ============================================================
# Self-test / demonstration
# ============================================================
if __name__ == "__main__":
    np.random.seed(7)
    vocab_size, seq_len = 24, 10
    model = ModernTransformer(vocab_size)

    print("=" * 60)
    print("Modern (2026) transformer — forward pass & generation")
    print("=" * 60)
    print(f"\nParameters: {model.num_parameters():,} "
          "(same ballpark as the Block 7 model — but the modern recipe)")

    # --- Shape check -------------------------------------------
    ids = list(np.random.randint(0, vocab_size, seq_len))
    logits = model.forward(ids)
    print(f"\n1) forward([{seq_len} tokens]) -> logits {logits.shape}")
    assert logits.shape == (seq_len, vocab_size)

    # --- Cache consistency across the whole stack --------------
    model.reset_cache()
    step = [model.forward([t], use_cache=True) for t in ids]
    cached_logits = np.concatenate(step, axis=0)
    err = np.abs(logits - cached_logits).max()
    print(f"2) full pass vs token-by-token KV-cached pass: "
          f"max diff = {err:.2e}")
    assert err < 1e-9

    # --- Generation (untrained — output is random by design) ---
    out = model.generate(prompt_ids=ids[:3], n_tokens=8, temperature=1.0)
    print(f"3) generate(): {ids[:3]} -> {out}")
    print("   (untrained weights => random tokens; the MECHANICS are")
    print("    what matter: prefill once, then one cheap step per token)")

    # --- The 2017 vs 2026 scorecard ----------------------------
    print("\n" + "=" * 60)
    print("What you've now built, vs what runs in production")
    print("=" * 60)
    rows = [
        ("Positions", "sinusoidal, added (Block 2)", "RoPE, rotated (Block 9)"),
        ("Norm", "LayerNorm (Block 5)", "RMSNorm (Block 10)"),
        ("FFN", "ReLU + biases (Block 5)", "SwiGLU, no biases (Block 10)"),
        ("Attention", "MHA (Block 4)", "GQA + KV cache (Block 11)"),
        ("Output", "separate Linear (Block 7)", "tied embeddings (Block 12)"),
        ("Scale", "~14K params", "~10^9 - 10^12 params"),
    ]
    print(f"   {'':<11}{'2017 / Blocks 1-8':<30}{'2026 / Blocks 9-12'}")
    for name, old, new in rows:
        print(f"   {name:<11}{old:<30}{new}")

    print("\nNext steps: read TRANSFORMERS_2026.md for the full story,")
    print("then try the exercises (add MoE routing, QK-Norm, ...).")
