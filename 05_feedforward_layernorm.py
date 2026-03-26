"""
============================================================
Block 5: Feed-Forward Network + Layer Normalization
============================================================
Every transformer block has two sub-layers:
  1. Multi-Head Attention (Block 4)
  2. Feed-Forward Network (this file)

Both are wrapped with Layer Normalization + residual connections,
which we also implement here.

Feed-Forward Network (FFN):
  A simple two-layer MLP applied to each token independently:
    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

  The hidden dimension (d_ff) is typically 4× the model dimension.
  This gives the model capacity to learn complex transformations.

  For our config: d_model=32 → d_ff=128

Layer Normalization:
  Normalizes across the feature dimension (d_model), making
  training much more stable. For each token position:
    mean = average of all 32 features
    std  = standard deviation of all 32 features
    normalized = (x - mean) / (std + ε)
    output = γ * normalized + β    (learnable scale & shift)

  This keeps activations in a reasonable range even as the
  network gets deeper.

We also implement a reusable Linear layer (matrix multiply + bias).
============================================================
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_module import load
_b2 = load("02_embeddings.py")
xavier_init = _b2.xavier_init


class Linear:
    """
    A single fully-connected (dense) layer.

    Computes: output = input @ W + b

    This is the fundamental building block of neural networks.
    Each neuron computes a weighted sum of its inputs plus a bias.

    Args:
        in_features:  Input dimension
        out_features: Output dimension
    """

    def __init__(self, in_features, out_features, seed=42):
        self.W = xavier_init((in_features, out_features), seed=seed)
        self.b = np.zeros(out_features)

        # Gradients
        self.grad_W = None
        self.grad_b = None

        # Cache
        self._input = None

    def forward(self, X):
        """
        Args:
            X: shape (seq_len, in_features)
        Returns:
            shape (seq_len, out_features)
        """
        self._input = X
        return X @ self.W + self.b

    def backward(self, grad_output):
        """
        Args:
            grad_output: shape (seq_len, out_features)
        Returns:
            grad_input: shape (seq_len, in_features)
        """
        X = self._input

        # Gradient w.r.t. weights: X^T @ grad_output
        self.grad_W = X.T @ grad_output

        # Gradient w.r.t. bias: sum over seq_len dimension
        self.grad_b = np.sum(grad_output, axis=0)

        # Gradient w.r.t. input: grad_output @ W^T
        grad_input = grad_output @ self.W.T
        return grad_input

    def parameters(self):
        return [(self.W, self.grad_W), (self.b, self.grad_b)]


class ReLU:
    """
    Rectified Linear Unit activation function.

    ReLU(x) = max(0, x)

    Simple, effective, and the most common activation function
    in deep learning. It introduces non-linearity: without it,
    stacking linear layers would just produce another linear layer.

    Gradient:
      If x > 0: gradient = 1 (pass through)
      If x ≤ 0: gradient = 0 (block)
    """

    def __init__(self):
        self._mask = None

    def forward(self, X):
        self._mask = (X > 0).astype(float)  # 1 where positive, 0 where negative
        return X * self._mask

    def backward(self, grad_output):
        return grad_output * self._mask      # zero out gradients where input was ≤ 0

    def parameters(self):
        return []  # no learnable parameters


class FeedForward:
    """
    Position-wise Feed-Forward Network.

    FFN(x) = Linear_2(ReLU(Linear_1(x)))

    Applied independently to each token position.
    Expands from d_model → d_ff, then compresses back:
      (seq_len, 32) → (seq_len, 128) → (seq_len, 32)

    This is where the model does most of its "thinking" —
    the attention layer routes information between positions,
    and the FFN processes information within each position.
    """

    def __init__(self, d_model, d_ff, seed=42):
        self.linear1 = Linear(d_model, d_ff, seed=seed)
        self.relu = ReLU()
        self.linear2 = Linear(d_ff, d_model, seed=seed + 10)

    def forward(self, X):
        """
        Args:
            X: shape (seq_len, d_model)
        Returns:
            shape (seq_len, d_model)
        """
        h = self.linear1.forward(X)       # (seq_len, d_ff)
        h = self.relu.forward(h)          # (seq_len, d_ff)  — zeros out negatives
        h = self.linear2.forward(h)       # (seq_len, d_model)
        return h

    def backward(self, grad_output):
        """
        Chain rule in reverse order.
        """
        grad = self.linear2.backward(grad_output)
        grad = self.relu.backward(grad)
        grad = self.linear1.backward(grad)
        return grad

    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()


class LayerNorm:
    """
    Layer Normalization.

    For each token position, normalize across the feature dimension:
      1. Compute mean (μ) and variance (σ²) of the d_model features
      2. Normalize: x_hat = (x - μ) / √(σ² + ε)
      3. Scale and shift: output = γ * x_hat + β

    γ (gamma) and β (beta) are learnable parameters that let the
    model undo the normalization if needed. They start at 1 and 0
    respectively (identity transformation).

    ε (epsilon) = 1e-5 prevents division by zero.

    Why layer norm?
      - Keeps activations in a stable range → easier training
      - Applied per-token (not per-batch like BatchNorm)
      - Critical for transformer training stability
    """

    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(d_model)     # scale (starts at 1)
        self.beta = np.zeros(d_model)     # shift (starts at 0)

        # Gradients
        self.grad_gamma = None
        self.grad_beta = None

        # Cache
        self._cache = None

    def forward(self, X):
        """
        Args:
            X: shape (seq_len, d_model)
        Returns:
            shape (seq_len, d_model)
        """
        # Compute statistics per token (across feature dimension)
        mean = np.mean(X, axis=-1, keepdims=True)          # (seq_len, 1)
        var = np.var(X, axis=-1, keepdims=True)             # (seq_len, 1)

        # Normalize
        x_hat = (X - mean) / np.sqrt(var + self.eps)       # (seq_len, d_model)

        # Scale and shift
        output = self.gamma * x_hat + self.beta             # (seq_len, d_model)

        # Save for backward
        self._cache = {
            "X": X, "mean": mean, "var": var, "x_hat": x_hat
        }
        return output

    def backward(self, grad_output):
        """
        Backward pass through layer normalization.

        This is the trickiest backward in the transformer because
        mean and variance depend on ALL features, creating complex
        dependencies in the gradient computation.

        Args:
            grad_output: shape (seq_len, d_model)
        Returns:
            grad_input: shape (seq_len, d_model)
        """
        cache = self._cache
        x_hat = cache["x_hat"]
        var = cache["var"]
        X = cache["X"]
        mean = cache["mean"]
        N = self.d_model

        # Gradients for learnable parameters
        self.grad_gamma = np.sum(grad_output * x_hat, axis=0)
        self.grad_beta = np.sum(grad_output, axis=0)

        # Gradient w.r.t. x_hat
        dx_hat = grad_output * self.gamma

        # Gradient w.r.t. input X (using the layer norm backward formula)
        std_inv = 1.0 / np.sqrt(var + self.eps)

        # Three terms contribute to grad_X:
        # 1. Direct: dx_hat * std_inv
        # 2. Through mean: -std_inv * mean(dx_hat)
        # 3. Through variance: -x_hat * mean(dx_hat * x_hat)
        grad_X = (1.0 / N) * std_inv * (
            N * dx_hat
            - np.sum(dx_hat, axis=-1, keepdims=True)
            - x_hat * np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
        )

        return grad_X

    def parameters(self):
        return [(self.gamma, self.grad_gamma), (self.beta, self.grad_beta)]


# ─── Self-test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Block 5: Feed-Forward Network + Layer Norm")
    print("=" * 55)

    np.random.seed(42)
    d_model = 32
    d_ff = 128
    seq_len = 6

    X = np.random.randn(seq_len, d_model)

    # 1. Linear layer
    print("\n--- Linear Layer ---")
    lin = Linear(d_model, d_ff, seed=42)
    out = lin.forward(X)
    print(f"  Input:  {X.shape} → Output: {out.shape}")
    print(f"  Params: W {lin.W.shape}, b {lin.b.shape}")

    # 2. ReLU
    print("\n--- ReLU ---")
    relu = ReLU()
    relu_out = relu.forward(out)
    neg_count = np.sum(out < 0)
    zero_count = np.sum(relu_out == 0)
    print(f"  Negative values in input: {neg_count}")
    print(f"  Zeros in output: {zero_count}")
    print(f"  (ReLU zeroed out all negative values)")

    # 3. Feed-Forward Network
    print("\n--- Feed-Forward Network ---")
    ffn = FeedForward(d_model, d_ff, seed=42)
    ffn_out = ffn.forward(X)
    print(f"  Input:  {X.shape}")
    print(f"  Hidden: (seq_len, {d_ff})  ← expanded 4×")
    print(f"  Output: {ffn_out.shape}  ← back to d_model")
    total_params = sum(p.size for p, _ in ffn.parameters())
    print(f"  FFN parameters: {total_params:,}")

    # 4. Layer Normalization
    print("\n--- Layer Normalization ---")
    ln = LayerNorm(d_model)

    print(f"\n  Before LayerNorm:")
    print(f"    Mean per token:  {np.mean(X, axis=-1).round(3)}")
    print(f"    Std per token:   {np.std(X, axis=-1).round(3)}")

    ln_out = ln.forward(X)
    print(f"\n  After LayerNorm:")
    print(f"    Mean per token:  {np.mean(ln_out, axis=-1).round(3)}")
    print(f"    Std per token:   {np.std(ln_out, axis=-1).round(3)}")
    print(f"  → Mean ≈ 0, Std ≈ 1 for each token position!")

    # 5. Backward pass test
    print("\n--- Backward Passes ---")
    fake_grad = np.random.randn(seq_len, d_model) * 0.01

    # FFN backward
    grad_x_ffn = ffn.backward(fake_grad)
    print(f"  FFN backward: grad_X shape = {grad_x_ffn.shape} ✓")

    # LayerNorm backward
    grad_x_ln = ln.backward(fake_grad)
    print(f"  LayerNorm backward: grad_X shape = {grad_x_ln.shape} ✓")

    # 6. Gradient check for LayerNorm
    print("\n--- Gradient Check (LayerNorm, gamma[0]) ---")
    eps = 1e-5
    loss_fn = lambda out: np.sum(fake_grad * out)

    orig = ln.gamma[0]
    ln.gamma[0] = orig + eps
    out_plus = ln.forward(X)
    ln.gamma[0] = orig - eps
    out_minus = ln.forward(X)
    ln.gamma[0] = orig
    ln.forward(X)  # restore cache
    ln.backward(fake_grad)

    numerical = (loss_fn(out_plus) - loss_fn(out_minus)) / (2 * eps)
    analytical = ln.grad_gamma[0]
    print(f"  Numerical:  {numerical:.6f}")
    print(f"  Analytical: {analytical:.6f}")
    print(f"  Difference: {abs(numerical - analytical):.2e}")
    print(f"  {'✓ Match!' if abs(numerical - analytical) < 1e-4 else '✗ Mismatch!'}")
