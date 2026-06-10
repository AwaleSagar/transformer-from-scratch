"""
============================================================
Block 10: RMSNorm + SwiGLU
============================================================
Two quiet but universal upgrades. Compare the 2017 stack you
built in Block 5 with what every major model uses in 2026:

  2017 (Blocks 5-8)            2026 (Llama/Qwen/Gemma/...)
  -----------------            ---------------------------
  LayerNorm                 →  RMSNorm
  FFN: ReLU(xW1+b1)W2+b2    →  SwiGLU: (SiLU(xW_g) ⊙ xW_u)W_d
  biases everywhere         →  almost no bias terms

------------------------------------------------------------
RMSNorm (Zhang & Sennrich, 2019)
------------------------------------------------------------
LayerNorm:  y = γ · (x − μ) / √(σ² + ε) + β
RMSNorm:    y = γ · x / RMS(x),  RMS(x) = √(mean(x²) + ε)

RMSNorm skips the mean subtraction and the β shift. That's
it. It's cheaper, has half the parameters, and trains just
as well — so everyone switched. (Some models also normalize
queries and keys inside attention — "QK-Norm" — using the
same RMSNorm formula to stabilize training.)

------------------------------------------------------------
SwiGLU (Shazeer, 2020, "GLU Variants Improve Transformer")
------------------------------------------------------------
Old FFN:    hidden = ReLU(x @ W1 + b1)        # one path
            out    = hidden @ W2 + b2

SwiGLU FFN: gate   = SiLU(x @ W_gate)          # "how much?"
            up     = x @ W_up                  # "what?"
            out    = (gate * up) @ W_down      # gated, then project

SiLU(x) = x · sigmoid(x) — a smooth ReLU that never has an
exactly-zero gradient. The element-wise product lets the
network learn to GATE information per dimension, which turns
out to be worth the third weight matrix. To keep parameter
count comparable, the hidden size shrinks from 4·d to ~8/3·d.
============================================================
"""

import numpy as np


def silu(x):
    """SiLU / Swish activation: x * sigmoid(x). Smooth ReLU."""
    return x / (1.0 + np.exp(-x))


class RMSNorm:
    """Root-Mean-Square normalization. LayerNorm minus the mean/shift."""

    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)   # learnable scale (no beta!)
        self.eps = eps

    def forward(self, x):
        """
        Args:    x: (seq_len, d_model)
        Returns: normalized array, same shape
        """
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return self.gamma * x / rms


class SwiGLU:
    """The modern gated feed-forward network (no biases)."""

    def __init__(self, d_model, d_ff, seed=0):
        rng = np.random.default_rng(seed)
        s = np.sqrt(2.0 / (d_model + d_ff))
        self.W_gate = rng.normal(0, s, (d_model, d_ff))
        self.W_up = rng.normal(0, s, (d_model, d_ff))
        self.W_down = rng.normal(0, s, (d_ff, d_model))

    def forward(self, x):
        """
        Args:    x: (seq_len, d_model)
        Returns: (seq_len, d_model)
        """
        gate = silu(x @ self.W_gate)      # decides how much passes
        up = x @ self.W_up                # the content being gated
        return (gate * up) @ self.W_down  # element-wise gate, project back


# ============================================================
# Self-test / demonstration
# ============================================================
if __name__ == "__main__":
    np.random.seed(0)
    d_model = 32
    x = np.random.randn(6, d_model) * 3 + 1.5   # messy activations

    print("=" * 60)
    print("RMSNorm vs LayerNorm")
    print("=" * 60)
    rms_norm = RMSNorm(d_model)
    y = rms_norm.forward(x)
    print(f"  input  : mean={x.mean():+.3f}  rms={np.sqrt((x**2).mean()):.3f}")
    print(f"  output : mean={y.mean():+.3f}  rms={np.sqrt((y**2).mean()):.3f}")
    print("  Note: RMS is pulled to ~1 but the mean is NOT forced to 0 —")
    print("  unlike LayerNorm. Models train fine without recentering.")

    # Parameter comparison
    layernorm_params = 2 * d_model     # gamma + beta
    rmsnorm_params = d_model           # gamma only
    print(f"\n  params per norm layer: LayerNorm={layernorm_params}, "
          f"RMSNorm={rmsnorm_params}  (50% fewer)")

    print()
    print("=" * 60)
    print("SwiGLU vs ReLU FFN")
    print("=" * 60)
    # Match parameter budgets: ReLU-FFN uses 2 matrices of d×4d;
    # SwiGLU uses 3 matrices, so hidden dim ~ 8/3·d keeps it equal.
    d_ff_relu = 4 * d_model
    d_ff_swiglu = int(8 * d_model / 3)
    relu_params = 2 * d_model * d_ff_relu
    swiglu_params = 3 * d_model * d_ff_swiglu
    print(f"  ReLU FFN  : 2 matrices, hidden={d_ff_relu:3d} -> {relu_params:,} params")
    print(f"  SwiGLU    : 3 matrices, hidden={d_ff_swiglu:3d} -> {swiglu_params:,} params")

    ffn = SwiGLU(d_model, d_ff_swiglu)
    out = ffn.forward(x)
    print(f"  output shape: {out.shape} (same as input, as required)")

    # Show the gating in action: a near-zero gate kills that channel
    print("\n  Gating demo (first token, first 6 hidden channels):")
    gate = silu(x @ ffn.W_gate)[0, :6]
    up = (x @ ffn.W_up)[0, :6]
    print(f"   gate : {gate.round(3)}")
    print(f"   up   : {up.round(3)}")
    print(f"   g*u  : {(gate * up).round(3)}   <- gate scales content per-channel")

    print("\nThese two swaps (plus dropping biases) are most of the")
    print("difference between GPT-2-era blocks and Llama-era blocks.")
