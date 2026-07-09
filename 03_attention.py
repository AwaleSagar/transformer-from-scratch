"""
============================================================
Block 3: Scaled Dot-Product Attention
============================================================
Attention is the core mechanism of the transformer. It lets
each token "look at" every other token and decide how much
to focus on each one.

Given three matrices:
  Q (Query)  — "what am I looking for?"
  K (Key)    — "what do I contain?"
  V (Value)  — "what information do I provide?"

The attention formula is:

  Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

Steps:
  1. Compute scores: how much each query matches each key
     scores = Q @ K^T  →  shape (seq_len, seq_len)

  2. Scale: divide by √d_k to prevent scores from getting
     too large (which would make softmax too "peaky")

  3. Mask (optional): for autoregressive (causal) models,
     set future positions to -∞ so they get 0 after softmax

  4. Softmax: convert scores to probabilities (each row sums to 1)

  5. Weighted sum: multiply attention weights by V
     output = weights @ V  →  shape (seq_len, d_k)
============================================================
"""

import numpy as np


def softmax(x):
    """
    Compute softmax along the last axis.

    softmax(x_i) = exp(x_i) / Σ exp(x_j)

    Trick: subtract max(x) before exp() for numerical stability.
    This doesn't change the result (it cancels out) but prevents
    overflow when x contains large values.

    Args:
        x: numpy array of any shape. Softmax is applied along
           the last dimension (axis=-1).

    Returns:
        Array of same shape with values in (0, 1) that sum to 1
        along the last axis.
    """
    # Subtract max for stability (per row)
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def create_causal_mask(seq_len):
    """
    Create a causal (look-ahead) mask for autoregressive decoding.

    The mask is an upper-triangular matrix of -infinity:
      [[  0, -inf, -inf, -inf],
       [  0,    0, -inf, -inf],
       [  0,    0,    0, -inf],
       [  0,    0,    0,    0]]

    When added to attention scores before softmax, -inf positions
    become 0 probability — the model can't "see" future tokens.

    Args:
        seq_len (int): Sequence length.

    Returns:
        numpy array of shape (seq_len, seq_len).
    """
    # np.triu with k=1 gives the strict upper triangle (1s above diagonal)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    # Replace 1s with -inf, 0s stay as 0
    return mask * (-1e9)  # use -1e9 instead of -inf for numerical safety


def attention_forward(Q, K, V, mask=None):
    """
    Scaled dot-product attention (forward pass).

    Args:
        Q: Query matrix,  shape (seq_len, d_k)
        K: Key matrix,    shape (seq_len, d_k)
        V: Value matrix,  shape (seq_len, d_k)
        mask: Optional causal mask, shape (seq_len, seq_len)

    Returns:
        output: Attention output, shape (seq_len, d_k)
        cache: Dictionary of intermediate values needed for backward
    """
    d_k = Q.shape[-1]

    # Step 1: Compute raw attention scores
    #   scores[i][j] = how much token i attends to token j
    scores = Q @ K.T                          # (seq_len, seq_len)

    # Step 2: Scale by √d_k
    scores = scores / np.sqrt(d_k)

    # Step 3: Apply causal mask (optional)
    if mask is not None:
        scores = scores + mask                # -inf positions → 0 after softmax

    # Step 4: Softmax → attention weights (probabilities)
    weights = softmax(scores)                 # (seq_len, seq_len)

    # Step 5: Weighted sum of values
    output = weights @ V                      # (seq_len, d_k)

    # Save everything needed for backward pass
    cache = {
        "Q": Q, "K": K, "V": V,
        "scores": scores,
        "weights": weights,
        "d_k": d_k,
        "mask": mask,
    }
    return output, cache


def attention_backward(grad_output, cache):
    """
    Backward pass through scaled dot-product attention.

    This computes gradients for Q, K, and V given the gradient
    of the loss w.r.t. the attention output.

    The chain rule flows backward through:
      output = weights @ V
      weights = softmax(scores)
      scores = Q @ K^T / √d_k

    Args:
        grad_output: Gradient of loss w.r.t. attention output,
                     shape (seq_len, d_k)
        cache: Intermediate values from forward pass.

    Returns:
        grad_Q: shape (seq_len, d_k)
        grad_K: shape (seq_len, d_k)
        grad_V: shape (seq_len, d_k)
    """
    Q = cache["Q"]
    K = cache["K"]
    V = cache["V"]
    weights = cache["weights"]
    d_k = cache["d_k"]

    # ── Step 5 backward: output = weights @ V ──
    # grad_weights = grad_output @ V^T
    # grad_V = weights^T @ grad_output
    grad_weights = grad_output @ V.T          # (seq_len, seq_len)
    grad_V = weights.T @ grad_output          # (seq_len, d_k)

    # ── Step 4 backward: weights = softmax(scores) ──
    # Softmax backward: if y = softmax(x), then
    #   dy/dx = diag(y) - y·y^T
    # For each row i:
    #   grad_scores[i] = weights[i] * (grad_weights[i] - Σ_j(grad_weights[i,j] * weights[i,j]))
    #
    # Simplified: grad_scores = weights * (grad_weights - (grad_weights * weights).sum(axis=-1, keepdims=True))
    sum_term = np.sum(grad_weights * weights, axis=-1, keepdims=True)
    grad_scores = weights * (grad_weights - sum_term)

    # ── Step 2 backward: scores = Q @ K^T / √d_k ──
    grad_scores = grad_scores / np.sqrt(d_k)

    # ── Step 1 backward: scores = Q @ K^T ──
    grad_Q = grad_scores @ K                  # (seq_len, d_k)
    grad_K = grad_scores.T @ Q                # (seq_len, d_k)

    return grad_Q, grad_K, grad_V


# ─── Self-test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Block 3: Scaled Dot-Product Attention")
    print("=" * 55)

    np.random.seed(42)
    seq_len = 4
    d_k = 8

    # Random Q, K, V inputs
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    # 1. Attention without mask
    print("\n--- Attention (no mask) ---")
    output, cache = attention_forward(Q, K, V)
    print(f"  Q shape: {Q.shape}")
    print(f"  K shape: {K.shape}")
    print(f"  V shape: {V.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"\n  Attention weights (each row sums to 1):")
    for i in range(seq_len):
        row = cache["weights"][i]
        print(f"    Token {i}: [{', '.join(f'{w:.3f}' for w in row)}]  sum={row.sum():.3f}")

    # 2. Attention with causal mask
    print("\n--- Attention (causal mask) ---")
    mask = create_causal_mask(seq_len)
    output_masked, cache_masked = attention_forward(Q, K, V, mask=mask)
    print(f"  Causal mask:")
    for i in range(seq_len):
        row = mask[i]
        print(f"    [{', '.join(f'{v:10.1f}' for v in row)}]")

    print(f"\n  Masked attention weights:")
    for i in range(seq_len):
        row = cache_masked["weights"][i]
        print(f"    Token {i}: [{', '.join(f'{w:.3f}' for w in row)}]  sum={row.sum():.3f}")
    print(f"  → Notice: each token only attends to itself and earlier tokens!")

    # 3. Backward pass
    print("\n--- Backward Pass ---")
    fake_grad = np.ones_like(output) * 0.1
    grad_Q, grad_K, grad_V = attention_backward(fake_grad, cache)
    print(f"  grad_Q shape: {grad_Q.shape}")
    print(f"  grad_K shape: {grad_K.shape}")
    print(f"  grad_V shape: {grad_V.shape}")
    print(f"  grad_Q[0, :4]: {grad_Q[0, :4].round(4)}")

    # 4. Numerical gradient check (verify backward is correct)
    print("\n--- Gradient Check (numerical vs analytical) ---")
    eps = 1e-5
    # Check one element of grad_Q
    i, j = 0, 0
    Q_plus = Q.copy(); Q_plus[i, j] += eps
    Q_minus = Q.copy(); Q_minus[i, j] -= eps
    out_plus, _ = attention_forward(Q_plus, K, V)
    out_minus, _ = attention_forward(Q_minus, K, V)
    numerical_grad = np.sum(fake_grad * (out_plus - out_minus)) / (2 * eps)
    analytical_grad = grad_Q[i, j]
    print(f"  Q[{i},{j}] numerical grad:  {numerical_grad:.6f}")
    print(f"  Q[{i},{j}] analytical grad: {analytical_grad:.6f}")
    print(f"  Difference: {abs(numerical_grad - analytical_grad):.2e}")
    print(f"  {'✓ Gradients match!' if abs(numerical_grad - analytical_grad) < 1e-5 else '✗ Mismatch!'}")
    # Gate: fail loudly if the analytical gradient is wrong (not just a print).
    assert abs(numerical_grad - analytical_grad) < 1e-5, "Gradient check failed!"
