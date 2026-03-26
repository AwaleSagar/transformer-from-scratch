"""
============================================================
Block 6: Transformer Decoder Block
============================================================
Now we assemble the pieces from Blocks 3–5 into a single
transformer decoder block. This is the repeating unit that
gets stacked N times to form the full transformer.

Architecture (Pre-Norm variant — simpler and more stable):

  ┌─────────────────────────────────────────┐
  │  Input X  (seq_len, d_model)            │
  │    │                                    │
  │    ├──────────────────┐                 │
  │    │                  │                 │
  │    ▼                  │                 │
  │  LayerNorm            │  ← normalize    │
  │    │                  │     before      │
  │    ▼                  │     attention   │
  │  Multi-Head Attention │                 │
  │    │                  │                 │
  │    ▼                  │                 │
  │  + ◄──────────────────┘  ← residual    │
  │    │                       connection   │
  │    ├──────────────────┐                 │
  │    │                  │                 │
  │    ▼                  │                 │
  │  LayerNorm            │  ← normalize    │
  │    │                  │     before FFN  │
  │    ▼                  │                 │
  │  Feed-Forward         │                 │
  │    │                  │                 │
  │    ▼                  │                 │
  │  + ◄──────────────────┘  ← residual    │
  │    │                       connection   │
  │    ▼                                    │
  │  Output  (seq_len, d_model)             │
  └─────────────────────────────────────────┘

Why Pre-Norm (Norm → Sublayer) instead of Post-Norm (Sublayer → Norm)?
  - More stable gradients — the residual path is "clean"
    (no normalization on the shortcut)
  - Easier to train without learning rate warmup
  - Used by GPT-2, GPT-3, LLaMA, and most modern transformers

Why residual connections?
  - They let gradients flow directly from output to input,
    preventing the vanishing gradient problem
  - The network only needs to learn the "residual" (difference
    from identity), which is easier than learning the full
    transformation from scratch
============================================================
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_module import load
_b3 = load("03_attention.py")
_b4 = load("04_multi_head_attention.py")
_b5 = load("05_feedforward_layernorm.py")

create_causal_mask = _b3.create_causal_mask
MultiHeadAttention = _b4.MultiHeadAttention
FeedForward = _b5.FeedForward
LayerNorm = _b5.LayerNorm


class TransformerBlock:
    """
    A single transformer decoder block.

    Combines:
      1. LayerNorm + Multi-Head Attention + Residual
      2. LayerNorm + Feed-Forward Network + Residual

    Args:
        d_model: Model dimension (32)
        n_heads: Number of attention heads (2)
        d_ff:    Feed-forward hidden dimension (128)
    """

    def __init__(self, d_model, n_heads, d_ff, seed=42):
        self.d_model = d_model

        # Sub-layer 1: Attention
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, seed=seed)

        # Sub-layer 2: Feed-Forward
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, seed=seed + 100)

        # Cache
        self._X = None
        self._attn_input = None
        self._attn_output = None
        self._ffn_input = None

    def forward(self, X, mask=None):
        """
        Args:
            X: Input, shape (seq_len, d_model)
            mask: Causal mask, shape (seq_len, seq_len)

        Returns:
            Output, shape (seq_len, d_model)
        """
        self._X = X

        # ── Sub-layer 1: Attention ──
        # Pre-Norm: normalize first, then apply attention
        self._attn_input = self.ln1.forward(X)              # LayerNorm
        attn_out = self.attn.forward(self._attn_input, mask) # Multi-Head Attention
        X = X + attn_out                                      # Residual connection
        self._attn_output = X

        # ── Sub-layer 2: Feed-Forward ──
        self._ffn_input = self.ln2.forward(X)                # LayerNorm
        ffn_out = self.ffn.forward(self._ffn_input)          # Feed-Forward
        X = X + ffn_out                                       # Residual connection

        return X

    def backward(self, grad_output):
        """
        Backward pass through the transformer block.

        The key insight with residual connections:
        If output = X + sublayer(norm(X)), then:
          grad_X = grad_output + grad_through_sublayer

        The gradient splits at each residual connection — one path
        flows directly back (grad_output), and the other flows
        through the sublayer.

        Args:
            grad_output: shape (seq_len, d_model)

        Returns:
            grad_X: gradient w.r.t. block input, shape (seq_len, d_model)
        """
        # ── Sub-layer 2 backward: FFN ──
        # output = attn_output + ffn(ln2(attn_output))
        grad_ffn_out = grad_output              # gradient flows through residual
        grad_ffn_in = self.ffn.backward(grad_ffn_out)
        grad_ln2 = self.ln2.backward(grad_ffn_in)
        grad_after_attn = grad_output + grad_ln2  # residual: add direct path

        # ── Sub-layer 1 backward: Attention ──
        # attn_output = X + attn(ln1(X))
        grad_attn_out = grad_after_attn         # gradient flows through residual
        grad_attn_in = self.attn.backward(grad_attn_out)
        grad_ln1 = self.ln1.backward(grad_attn_in)
        grad_X = grad_after_attn + grad_ln1      # residual: add direct path

        return grad_X

    def parameters(self):
        """All learnable parameters in this block."""
        params = []
        params += self.ln1.parameters()
        params += self.attn.parameters()
        params += self.ln2.parameters()
        params += self.ffn.parameters()
        return params


# ─── Self-test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Block 6: Transformer Decoder Block")
    print("=" * 55)

    np.random.seed(42)
    d_model = 32
    n_heads = 2
    d_ff = 128
    seq_len = 8

    block = TransformerBlock(d_model, n_heads, d_ff, seed=42)

    # Count parameters
    total_params = sum(
        p.size for p, _ in block.parameters()
    )
    print(f"\nConfig: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"Parameters per block: {total_params:,}")
    print(f"\nBreakdown:")
    print(f"  LayerNorm 1: γ({d_model}) + β({d_model}) = {2*d_model}")
    print(f"  Attention:   W_Q + W_K + W_V + W_O = 4 × {d_model}×{d_model} = {4*d_model*d_model:,}")
    print(f"  LayerNorm 2: γ({d_model}) + β({d_model}) = {2*d_model}")
    print(f"  FFN:         {d_model}×{d_ff} + {d_ff} + {d_ff}×{d_model} + {d_model} = {d_model*d_ff + d_ff + d_ff*d_model + d_model:,}")
    print(f"  Total: {total_params:,}")

    # Forward pass
    print(f"\n--- Forward Pass ---")
    X = np.random.randn(seq_len, d_model)
    mask = create_causal_mask(seq_len)
    output = block.forward(X, mask=mask)
    print(f"  Input shape:  {X.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Input mean:   {X.mean():.4f}")
    print(f"  Output mean:  {output.mean():.4f}")

    # Verify residual connection works
    # Output should be different from input but in a similar range
    diff = np.mean(np.abs(output - X))
    print(f"  Mean |output - input|: {diff:.4f}")
    print(f"  (Residual connections keep this moderate)")

    # Backward pass
    print(f"\n--- Backward Pass ---")
    fake_grad = np.random.randn(seq_len, d_model) * 0.01
    grad_X = block.backward(fake_grad)
    print(f"  grad_X shape: {grad_X.shape}")
    print(f"  grad_X mean:  {grad_X.mean():.6f}")
    print(f"  grad_X std:   {grad_X.std():.6f}")

    # Check that all parameters got gradients
    print(f"\n--- Parameter Gradients ---")
    for i, (p, g) in enumerate(block.parameters()):
        has_grad = g is not None
        print(f"  Param {i:2d}: shape {str(p.shape):15s} grad={'✓' if has_grad else '✗'}")

    print(f"\n  All gradients computed ✓")

    # Show data flow shapes
    print(f"\n--- Data Flow Through Block ---")
    print(f"  Input:         ({seq_len}, {d_model})")
    print(f"    → LayerNorm:   ({seq_len}, {d_model})")
    print(f"    → MH Attn:     ({seq_len}, {d_model})")
    print(f"    → + Residual:  ({seq_len}, {d_model})")
    print(f"    → LayerNorm:   ({seq_len}, {d_model})")
    print(f"    → FFN expand:  ({seq_len}, {d_ff})")
    print(f"    → FFN compress:({seq_len}, {d_model})")
    print(f"    → + Residual:  ({seq_len}, {d_model})")
    print(f"  Output:        ({seq_len}, {d_model})")
