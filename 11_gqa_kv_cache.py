"""
============================================================
Block 11: Grouped-Query Attention (GQA) + KV Cache
============================================================
These two ideas are about INFERENCE — making generation fast
and memory-cheap. They're why a 7B model can run on a laptop.

------------------------------------------------------------
The KV cache
------------------------------------------------------------
Generating token #100 the naive way (Block 8) re-runs the
whole model on all 100 tokens. But the keys and values of
tokens 1-99 haven't changed! So we CACHE them:

  step 1: compute K,V for token 1, store them
  step 2: compute Q,K,V for token 2 only;
          attend against cached K,V + new K,V; append to cache
  ...

Each step now costs O(1) model passes instead of O(n).
The price: the cache itself. Its size is

  2 (K and V) × layers × kv_heads × d_head × seq_len × bytes

For long contexts this becomes the dominant memory cost —
which motivates the next idea.

------------------------------------------------------------
Grouped-Query Attention (Ainslie et al., 2023)
------------------------------------------------------------
Multi-Head Attention (Block 4): every head has its own Q, K, V.
GQA: several query heads SHARE one K/V head.

  MHA:  8 Q-heads, 8 KV-heads   (cache: 8 units)
  GQA:  8 Q-heads, 2 KV-heads   (cache: 2 units — 4x smaller)
  MQA:  8 Q-heads, 1 KV-head    (extreme case)

Quality barely drops, KV-cache memory plummets. GQA is the
default in Llama 3, Qwen3, Gemma 3, Mistral, gpt-oss...
(DeepSeek goes further with Multi-Head Latent Attention,
which compresses K/V into a small latent vector.)
============================================================
"""

import numpy as np
from load_module import load

_block3 = load("03_attention.py")
softmax = _block3.softmax

_block9 = load("09_rope.py")
rope_frequencies = _block9.rope_frequencies
apply_rope = _block9.apply_rope


class GroupedQueryAttention:
    """
    Causal self-attention with n_q_heads query heads sharing
    n_kv_heads key/value heads, RoPE, and an optional KV cache.
    """

    def __init__(self, d_model, n_q_heads, n_kv_heads, max_seq_len=64, seed=0):
        assert d_model % n_q_heads == 0
        assert n_q_heads % n_kv_heads == 0
        self.n_q = n_q_heads
        self.n_kv = n_kv_heads
        self.group = n_q_heads // n_kv_heads      # q-heads per kv-head
        self.d_head = d_model // n_q_heads

        rng = np.random.default_rng(seed)
        s = np.sqrt(2.0 / (2 * d_model))
        self.W_Q = rng.normal(0, s, (d_model, n_q_heads * self.d_head))
        self.W_K = rng.normal(0, s, (d_model, n_kv_heads * self.d_head))  # smaller!
        self.W_V = rng.normal(0, s, (d_model, n_kv_heads * self.d_head))  # smaller!
        self.W_O = rng.normal(0, s, (n_q_heads * self.d_head, d_model))

        self.cos, self.sin = rope_frequencies(self.d_head, max_seq_len)
        self.reset_cache()

    def reset_cache(self):
        """Empty the KV cache (start of a new sequence)."""
        self.cache_k = np.zeros((self.n_kv, 0, self.d_head))
        self.cache_v = np.zeros((self.n_kv, 0, self.d_head))

    def forward(self, x, use_cache=False):
        """
        Args:
            x: (seq_len, d_model) — with use_cache=True during
               generation, pass ONLY the new token(s).
            use_cache: append K/V to the cache and attend over
               the full cached history.

        Returns: (seq_len, d_model)
        """
        seq_len = x.shape[0]
        past = self.cache_k.shape[1] if use_cache else 0

        # Project. Note K/V are (n_kv) heads — fewer than Q.
        Q = (x @ self.W_Q).reshape(seq_len, self.n_q, self.d_head).transpose(1, 0, 2)
        K = (x @ self.W_K).reshape(seq_len, self.n_kv, self.d_head).transpose(1, 0, 2)
        V = (x @ self.W_V).reshape(seq_len, self.n_kv, self.d_head).transpose(1, 0, 2)

        # RoPE: rotate by ABSOLUTE position (past + offset),
        # so cached and new tokens line up correctly.
        for h in range(self.n_q):
            Q[h] = apply_rope_at(Q[h], self.cos, self.sin, start=past)
        for h in range(self.n_kv):
            K[h] = apply_rope_at(K[h], self.cos, self.sin, start=past)

        if use_cache:
            self.cache_k = np.concatenate([self.cache_k, K], axis=1)
            self.cache_v = np.concatenate([self.cache_v, V], axis=1)
            K, V = self.cache_k, self.cache_v       # full history

        total = K.shape[1]
        out_heads = []
        for h in range(self.n_q):
            kv = h // self.group                     # which kv-head serves me
            scores = Q[h] @ K[kv].T / np.sqrt(self.d_head)  # (seq_len, total)
            # Causal mask: new token i (absolute pos past+i) may
            # see absolute positions 0 .. past+i
            mask = np.arange(total)[None, :] > (past + np.arange(seq_len))[:, None]
            scores = np.where(mask, -1e9, scores)
            out_heads.append(softmax(scores) @ V[kv])  # (seq_len, d_head)

        concat = np.stack(out_heads, axis=1).reshape(seq_len, -1)
        return concat @ self.W_O


def apply_rope_at(x, cos, sin, start=0):
    """Apply RoPE with positions start, start+1, ... (for cached generation)."""
    n = x.shape[0]
    return apply_rope_slice(x, cos[start:start + n], sin[start:start + n])


def apply_rope_slice(x, cos, sin):
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out


# ============================================================
# Self-test / demonstration
# ============================================================
if __name__ == "__main__":
    np.random.seed(1)
    d_model, n_q, n_kv, seq_len = 32, 4, 2, 10
    x = np.random.randn(seq_len, d_model)

    print("=" * 60)
    print(f"GQA: {n_q} query heads sharing {n_kv} kv heads")
    print("=" * 60)

    # --- Check: cached generation == full forward pass --------
    attn = GroupedQueryAttention(d_model, n_q, n_kv)
    full = attn.forward(x)                        # all at once, no cache

    attn.reset_cache()
    step_outputs = []
    for t in range(seq_len):                      # one token at a time
        step_outputs.append(attn.forward(x[t:t + 1], use_cache=True))
    cached = np.concatenate(step_outputs, axis=0)

    err = np.abs(full - cached).max()
    print(f"\n1) KV-cached generation matches full forward pass:")
    print(f"   max difference = {err:.2e}")
    assert err < 1e-9, "KV cache mismatch!"

    # --- Memory math -------------------------------------------
    print("\n2) Why GQA matters — KV cache size (fp16, per layer):")
    print(f"   {'config':<28}{'cache entries':<18}{'@8k ctx, 32 layers'}")
    d_head = d_model // n_q
    for name, kv_heads in [("MHA (every head has KV)", n_q),
                           ("GQA (shared KV)", n_kv),
                           ("MQA (single KV head)", 1)]:
        per_tok = 2 * kv_heads * d_head            # toy numbers
        # realistic example: d_head=128, 8k context, 32 layers, fp16
        real = 2 * kv_heads / n_q * 8 * 128 * 8192 * 32 * 2 / 1e9
        print(f"   {name:<28}{per_tok:<18}{real:.2f} GB")

    # --- Parameter savings -------------------------------------
    mha_kv_params = 2 * d_model * d_model
    gqa_kv_params = 2 * d_model * (n_kv * d_head)
    print(f"\n3) K+V projection params: MHA={mha_kv_params:,}  "
          f"GQA={gqa_kv_params:,}  ({n_q // n_kv}x smaller)")

    print("\nGQA + KV cache is how chat models stream tokens to you")
    print("in real time without re-reading the whole conversation.")
