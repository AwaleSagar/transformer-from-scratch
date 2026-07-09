"""
============================================================
Block 4: Multi-Head Attention
============================================================
Instead of performing one attention function, multi-head
attention runs several attention "heads" in parallel, each
looking at the input from a different perspective.

Think of it like this:
  - Head 1 might learn to focus on the *subject* of a sentence
  - Head 2 might learn to focus on the *verb*
  - Each head captures a different type of relationship

Architecture:
  1. Project input X into Q, K, V using learnable weight matrices
  2. Split Q, K, V into n_heads smaller chunks
  3. Run attention on each chunk independently
  4. Concatenate results from all heads
  5. Project concatenated output through W_O

Shapes (for our config: d_model=32, n_heads=2, d_k=16):
  Input X:    (seq_len, 32)
  W_Q, W_K, W_V: (32, 32) each
  Q, K, V:    (seq_len, 32)
  Per-head:   (seq_len, 16) × 2 heads
  Concat:     (seq_len, 32)
  W_O:        (32, 32)
  Output:     (seq_len, 32)
============================================================
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_module import load
_b2 = load("02_embeddings.py")
_b3 = load("03_attention.py")

xavier_init = _b2.xavier_init
attention_forward = _b3.attention_forward
attention_backward = _b3.attention_backward
create_causal_mask = _b3.create_causal_mask


class MultiHeadAttention:
    """
    Multi-Head Attention layer.

    Given input X of shape (seq_len, d_model):
      Q = X @ W_Q    (query projection)
      K = X @ W_K    (key projection)
      V = X @ W_V    (value projection)

    Then split into heads, run attention, concat, and project:
      output = Concat(head_1, ..., head_h) @ W_O
    """

    def __init__(self, d_model, n_heads, seed=42):
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head

        # Learnable projection matrices
        rng_offset = seed
        self.W_Q = xavier_init((d_model, d_model), seed=rng_offset)
        self.W_K = xavier_init((d_model, d_model), seed=rng_offset + 1)
        self.W_V = xavier_init((d_model, d_model), seed=rng_offset + 2)
        self.W_O = xavier_init((d_model, d_model), seed=rng_offset + 3)

        # Gradient accumulators (set during backward)
        self.grad_W_Q = None
        self.grad_W_K = None
        self.grad_W_V = None
        self.grad_W_O = None

        # Cache for backward
        self._cache = None

    def forward(self, X, mask=None):
        """
        Args:
            X: Input tensor, shape (seq_len, d_model)
            mask: Optional causal mask, shape (seq_len, seq_len)

        Returns:
            Output tensor, shape (seq_len, d_model)
        """
        seq_len = X.shape[0]

        # ── Step 1: Linear projections ──
        Q = X @ self.W_Q   # (seq_len, d_model)
        K = X @ self.W_K
        V = X @ self.W_V

        # ── Step 2: Split into heads ──
        # Reshape (seq_len, d_model) → (n_heads, seq_len, d_k)
        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        # ── Step 3: Attention per head ──
        head_outputs = []
        attn_caches = []
        for h in range(self.n_heads):
            head_out, attn_cache = attention_forward(
                Q_heads[h], K_heads[h], V_heads[h], mask=mask
            )
            head_outputs.append(head_out)   # each: (seq_len, d_k)
            attn_caches.append(attn_cache)

        # ── Step 4: Concatenate heads ──
        # Stack (seq_len, d_k) × n_heads → (seq_len, d_model)
        concat = np.concatenate(head_outputs, axis=-1)

        # ── Step 5: Output projection ──
        output = concat @ self.W_O   # (seq_len, d_model)

        # Save for backward
        self._cache = {
            "X": X,
            "Q": Q, "K": K, "V": V,
            "Q_heads": Q_heads,
            "K_heads": K_heads,
            "V_heads": V_heads,
            "head_outputs": head_outputs,
            "attn_caches": attn_caches,
            "concat": concat,
        }
        return output

    def backward(self, grad_output):
        """
        Backward pass through multi-head attention.

        Computes gradients for W_Q, W_K, W_V, W_O and
        the gradient w.r.t. input X.

        Args:
            grad_output: shape (seq_len, d_model)

        Returns:
            grad_X: gradient w.r.t. input, shape (seq_len, d_model)
        """
        cache = self._cache
        X = cache["X"]
        Q, K, V = cache["Q"], cache["K"], cache["V"]
        concat = cache["concat"]
        attn_caches = cache["attn_caches"]

        # ── Step 5 backward: output = concat @ W_O ──
        self.grad_W_O = concat.T @ grad_output       # (d_model, d_model)
        grad_concat = grad_output @ self.W_O.T        # (seq_len, d_model)

        # ── Step 4 backward: split grad_concat back into heads ──
        grad_head_outputs = self._split_heads(grad_concat)

        # ── Step 3 backward: attention per head ──
        grad_Q_heads = []
        grad_K_heads = []
        grad_V_heads = []
        for h in range(self.n_heads):
            gQ, gK, gV = attention_backward(
                grad_head_outputs[h], attn_caches[h]
            )
            grad_Q_heads.append(gQ)
            grad_K_heads.append(gK)
            grad_V_heads.append(gV)

        # ── Step 2 backward: merge heads ──
        grad_Q = self._merge_heads(grad_Q_heads)   # (seq_len, d_model)
        grad_K = self._merge_heads(grad_K_heads)
        grad_V = self._merge_heads(grad_V_heads)

        # ── Step 1 backward: Q = X @ W_Q, etc. ──
        self.grad_W_Q = X.T @ grad_Q
        self.grad_W_K = X.T @ grad_K
        self.grad_W_V = X.T @ grad_V

        # Gradient flows back through all three projections
        grad_X = grad_Q @ self.W_Q.T + grad_K @ self.W_K.T + grad_V @ self.W_V.T

        return grad_X

    def _split_heads(self, X):
        """
        Split (seq_len, d_model) → list of n_heads arrays of (seq_len, d_k).

        Example: d_model=32, n_heads=2, d_k=16
          X[:, 0:16]  → head 0
          X[:, 16:32] → head 1
        """
        heads = []
        for h in range(self.n_heads):
            start = h * self.d_k
            end = start + self.d_k
            heads.append(X[:, start:end])
        return heads

    def _merge_heads(self, head_list):
        """
        Merge list of n_heads arrays of (seq_len, d_k) → (seq_len, d_model).
        """
        return np.concatenate(head_list, axis=-1)

    def parameters(self):
        """Return list of (param, grad) tuples for the optimizer."""
        return [
            (self.W_Q, self.grad_W_Q),
            (self.W_K, self.grad_W_K),
            (self.W_V, self.grad_W_V),
            (self.W_O, self.grad_W_O),
        ]


# ─── Self-test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Block 4: Multi-Head Attention")
    print("=" * 55)

    np.random.seed(42)
    d_model = 32
    n_heads = 2
    seq_len = 6

    mha = MultiHeadAttention(d_model, n_heads, seed=42)
    print(f"\nConfig: d_model={d_model}, n_heads={n_heads}, d_k={mha.d_k}")
    total_params = sum(w.size for w, _ in mha.parameters())
    print(f"Total parameters: {total_params:,}")

    # Forward pass
    X = np.random.randn(seq_len, d_model)
    mask = create_causal_mask(seq_len)
    output = mha.forward(X, mask=mask)

    print(f"\n--- Forward Pass ---")
    print(f"  Input shape:  {X.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output[0, :8]: {output[0, :8].round(3)}")

    # Show attention weights per head
    print(f"\n--- Attention Weights Per Head ---")
    for h in range(n_heads):
        w = mha._cache["attn_caches"][h]["weights"]
        print(f"\n  Head {h} attention weights:")
        for i in range(seq_len):
            row = w[i]
            print(f"    Token {i}: [{', '.join(f'{v:.3f}' for v in row)}]")

    # Backward pass
    print(f"\n--- Backward Pass ---")
    fake_grad = np.random.randn(seq_len, d_model) * 0.01
    grad_X = mha.backward(fake_grad)
    print(f"  grad_X shape: {grad_X.shape}")
    print(f"  grad_W_Q shape: {mha.grad_W_Q.shape}")
    print(f"  grad_W_O shape: {mha.grad_W_O.shape}")

    # Numerical gradient check for W_Q[0,0]
    print(f"\n--- Gradient Check (W_Q[0,0]) ---")
    eps = 1e-5
    loss_fn = lambda out: np.sum(fake_grad * out)

    W_Q_orig = mha.W_Q[0, 0]
    mha.W_Q[0, 0] = W_Q_orig + eps
    out_plus = mha.forward(X, mask=mask)
    mha.W_Q[0, 0] = W_Q_orig - eps
    out_minus = mha.forward(X, mask=mask)
    mha.W_Q[0, 0] = W_Q_orig

    numerical = (loss_fn(out_plus) - loss_fn(out_minus)) / (2 * eps)
    analytical = mha.grad_W_Q[0, 0]
    print(f"  Numerical:  {numerical:.6f}")
    print(f"  Analytical: {analytical:.6f}")
    print(f"  Difference: {abs(numerical - analytical):.2e}")
    print(f"  {'✓ Match!' if abs(numerical - analytical) < 1e-4 else '✗ Mismatch!'}")
    # Gate: fail loudly if the analytical gradient is wrong (not just a print).
    assert abs(numerical - analytical) < 1e-4, "Gradient check failed!"
