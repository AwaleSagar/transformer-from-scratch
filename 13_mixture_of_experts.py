"""
============================================================
Block 13: Mixture-of-Experts (MoE)
============================================================
MODERN SERIES (2026): the first thing Block 12 said it was
"still missing." A dense transformer runs EVERY token through
the SAME feed-forward network. MoE replaces that one FFN with
MANY (the "experts") and routes each token to only its top-k
experts. You get the capacity of a huge FFN while paying the
compute of just a small one.

  Dense FFN (Block 12)          Mixture-of-Experts (this file)
  --------------------          ------------------------------
  1 SwiGLU, all tokens          N SwiGLU experts, top-k per token
  every param active            only k/N of the params active
  capacity == compute           capacity >> compute (sparse)

WHY IT MATTERS
DeepSeek-V3 has 671B parameters but activates only ~37B per
token. Mixtral, Qwen3-MoE, gpt-oss and friends all use this
"sparse" trick: scale total parameters without scaling the
per-token FLOPs.

HOW ROUTING WORKS
  1. A tiny linear "router" scores each token over the experts
         logits = x @ W_router        (seq_len, n_experts)
         probs  = softmax(logits)
  2. Each token picks its top-k experts (k=1 here by default).
  3. Only those experts run for that token; their outputs are
     combined, weighted by the (renormalized) routing probs.

LOAD BALANCING (printed, NOT trained here)
Left alone, the router can collapse — sending every token to
one favourite expert. Real systems add an auxiliary loss that
rewards spreading tokens evenly. We COMPUTE and PRINT it as a
diagnostic, but this block is forward-only (like Blocks 9-12),
so it is never added to a loss or backpropagated.

  aux = n_experts * Σ_i (f_i · P_i)
    f_i = fraction of tokens routed to expert i
    P_i = mean router probability for expert i

NOTE: forward pass only. We reuse the SwiGLU expert from
Block 10 and the softmax from Block 3 — no new math, just the
routing that turns one FFN into a sparse committee of them.
============================================================
"""

import numpy as np
from load_module import load

_b3 = load("03_attention.py")
softmax = _b3.softmax

_b10 = load("10_rmsnorm_swiglu.py")
SwiGLU = _b10.SwiGLU


class MoELayer:
    """
    Sparse Mixture-of-Experts feed-forward layer.

    A drop-in replacement for a dense SwiGLU FFN: same input and
    output shape, but each token is processed by only its top-k
    experts instead of one shared network.

    Args:
        d_model:   Model dimension.
        d_ff:      Hidden dimension of each expert (SwiGLU).
        n_experts: Number of expert FFNs.
        top_k:     Experts each token is routed to (1 = Switch-style).
        seed:      Base seed; router and each expert get distinct seeds.
    """

    def __init__(self, d_model, d_ff, n_experts=4, top_k=1, seed=0):
        assert 1 <= top_k <= n_experts, "top_k must be between 1 and n_experts"
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        # Router: a single linear layer, Xavier-style init (matches Block 10).
        rng = np.random.default_rng(seed)
        s = np.sqrt(2.0 / (d_model + n_experts))
        self.W_router = rng.normal(0, s, (d_model, n_experts))

        # The experts: independent SwiGLU FFNs (distinct seeds so they differ).
        self.experts = [SwiGLU(d_model, d_ff, seed=seed + 1 + e)
                        for e in range(n_experts)]

        # Populated by forward() for the demo / verify_all invariant check.
        self.last_mask = None    # (seq_len, n_experts) bool: who ran where
        self.last_probs = None   # (seq_len, n_experts) router probabilities
        self.last_aux = None     # scalar load-balancing diagnostic

    def forward(self, x):
        """
        Args:    x: (seq_len, d_model)
        Returns: (seq_len, d_model)
        """
        seq_len = x.shape[0]

        # 1. Route: score every token over the experts.
        logits = x @ self.W_router               # (seq_len, n_experts)
        probs = softmax(logits)                  # (seq_len, n_experts)

        # 2. Top-k selection per token -> boolean dispatch mask.
        #    argsort ascending; take the last top_k columns = highest probs.
        topk_idx = np.argsort(probs, axis=1)[:, -self.top_k:]   # (seq_len, top_k)
        mask = np.zeros_like(probs, dtype=bool)                 # (seq_len, n_experts)
        rows = np.arange(seq_len)[:, None]
        mask[rows, topk_idx] = True

        # 3. Combine weights: routing probs renormalized over the chosen experts
        #    (so each token's expert weights sum to 1; for top_k=1 this is 1.0).
        gate = np.where(mask, probs, 0.0)
        gate = gate / gate.sum(axis=1, keepdims=True)

        # 4. Dense dispatch: run each expert on the tokens routed to it, then
        #    accumulate its weighted output. (No token dropping / parallelism.)
        out = np.zeros_like(x)
        for e, expert in enumerate(self.experts):
            sel = mask[:, e]                     # tokens that picked expert e
            if not sel.any():
                continue
            y = expert.forward(x[sel])           # (n_sel, d_model)
            out[sel] += gate[sel, e][:, None] * y

        # 5. Load-balancing diagnostic (printed only — NOT backpropagated).
        f = mask.mean(axis=0)                    # fraction of tokens per expert
        P = probs.mean(axis=0)                   # mean router prob per expert
        self.last_aux = float(self.n_experts * np.sum(f * P))

        self.last_mask = mask
        self.last_probs = probs
        return out


# ============================================================
# Self-test / demonstration
# ============================================================
if __name__ == "__main__":
    np.random.seed(0)
    d_model, d_ff, seq_len = 32, int(8 * 32 / 3), 12
    n_experts, top_k = 4, 1

    print("=" * 60)
    print(f"Mixture-of-Experts: {n_experts} experts, top-{top_k} routing")
    print("=" * 60)

    moe = MoELayer(d_model, d_ff, n_experts=n_experts, top_k=top_k, seed=0)
    x = np.random.randn(seq_len, d_model)
    out = moe.forward(x)

    print(f"\n1) forward({x.shape}) -> {out.shape}  (same shape as input)")
    assert out.shape == x.shape

    # --- Routing histogram: how many tokens went to each expert -----
    counts = moe.last_mask.sum(axis=0)
    print(f"\n2) Routing histogram (tokens per expert):")
    for e in range(n_experts):
        bar = "█" * int(counts[e])
        print(f"   expert {e}: {int(counts[e]):2d}  {bar}")
    print("   (an untrained router may lean on a few experts — that's what")
    print("    the load-balancing loss below is designed to discourage)")

    # --- Load-balancing aux scalar (diagnostic only) ----------------
    print(f"\n3) Load-balancing aux = {moe.last_aux:.4f}")
    print("   (== 1.0 when perfectly balanced; up to n_experts as all tokens")
    print("    collapse onto a single expert — lower is better)")
    print("   NOT added to any loss and NOT backpropagated in this block.")

    # --- The invariant this block guarantees ------------------------
    active_per_token = moe.last_mask.sum(axis=1)
    print(f"\n4) Experts active per token: {active_per_token}")
    assert np.all(active_per_token == top_k), \
        "each token must be routed to exactly top_k experts!"
    print(f"   ✓ every token is routed to exactly top_k={top_k} expert(s)")

    print("\nThis is the last missing piece from Block 12's list: sparse")
    print("conditional compute. Scale the expert count up and you have the")
    print("recipe behind today's largest open models.")
