"""
============================================================
Block 9: Rotary Position Embeddings (RoPE)
============================================================
MODERN SERIES (2026): Blocks 9-12 upgrade the 2017-style
model from Blocks 1-8 to the architecture used by virtually
every modern LLM (Llama, Qwen, Gemma, DeepSeek, Mistral...).

WHAT CHANGED SINCE 2017?
The original transformer (and our Block 2) ADDS a position
vector to each token embedding. Modern models instead ROTATE
the query and key vectors inside attention — this is RoPE
(Su et al., 2021, "RoFormer").

WHY ROTATE?
Think of each pair of dimensions in Q/K as a point on a 2D
plane. RoPE rotates that point by an angle proportional to
the token's position:

  position 0 → rotate by 0°
  position 1 → rotate by θ
  position 2 → rotate by 2θ   ... and so on.

The magic: when you take the dot product of a rotated query
(position m) and a rotated key (position n), the result only
depends on the RELATIVE distance (m − n), not the absolute
positions. Attention naturally learns "3 tokens back" rather
than "position 7 looking at position 4".

Benefits over added sinusoidal encodings:
  - Relative positions are what language actually needs
  - Nothing is added to the embeddings (cleaner residual stream)
  - Extends to longer sequences far more gracefully
    (this is the basis of long-context tricks like YaRN)

Each pair of dims (2i, 2i+1) gets its own rotation speed:

  θ_i = 10000^(-2i / d)

Low dims rotate fast (capture nearby words), high dims rotate
slowly (capture far-apart structure) — same intuition as the
sine/cosine frequencies in Block 2, but applied as rotation.

NOTE: Blocks 9-12 focus on the FORWARD pass so the modern
ideas stay front and center. You already proved you can
hand-derive backward passes in Blocks 3-8.
============================================================
"""

import numpy as np


def rope_frequencies(d_head, max_seq_len, base=10000.0):
    """
    Precompute the rotation angles for every (position, dim-pair).

    Args:
        d_head: dimension of one attention head (must be even)
        max_seq_len: longest sequence we'll ever see
        base: frequency base (10000 in the original paper;
              modern long-context models raise this, e.g.
              Llama 3 uses 500000)

    Returns:
        cos, sin: arrays of shape (max_seq_len, d_head // 2)
    """
    assert d_head % 2 == 0, "head dimension must be even for RoPE"
    # One inverse frequency per dim-pair: 10000^(-2i/d)
    inv_freq = base ** (-np.arange(0, d_head, 2) / d_head)   # (d_head/2,)
    positions = np.arange(max_seq_len)                        # (seq_len,)
    # Outer product: angle for each (position, pair)
    angles = np.outer(positions, inv_freq)                    # (seq_len, d_head/2)
    return np.cos(angles), np.sin(angles)


def apply_rope(x, cos, sin):
    """
    Rotate each consecutive pair of dimensions of x by its
    position-dependent angle.

    For a pair (x1, x2) at position p with angle a = p * θ:
        x1' = x1·cos(a) − x2·sin(a)
        x2' = x1·sin(a) + x2·cos(a)
    (standard 2D rotation matrix)

    Args:
        x: (seq_len, d_head) — a query or key matrix for ONE head
        cos, sin: precomputed tables, sliced to (seq_len, d_head/2)

    Returns:
        Rotated array, same shape as x.
    """
    seq_len = x.shape[0]
    cos, sin = cos[:seq_len], sin[:seq_len]
    x1 = x[:, 0::2]   # even dims  (seq_len, d_head/2)
    x2 = x[:, 1::2]   # odd dims   (seq_len, d_head/2)
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out


# ============================================================
# Self-test / demonstration
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    d_head, seq_len = 8, 12
    cos, sin = rope_frequencies(d_head, max_seq_len=64)

    print("=" * 60)
    print("RoPE demo  (d_head=8, seq_len=12)")
    print("=" * 60)

    # --- Property 1: rotation preserves vector length ---------
    v = np.random.randn(seq_len, d_head)
    v_rot = apply_rope(v, cos, sin)
    print("\n1) Rotation preserves norms (no information destroyed):")
    print(f"   |v|     = {np.linalg.norm(v, axis=1)[:4].round(4)}")
    print(f"   |RoPE(v)| = {np.linalg.norm(v_rot, axis=1)[:4].round(4)}")

    # --- Property 2: dot product depends only on RELATIVE pos -
    # Put the SAME query vector at position 2 and the SAME key
    # vector at position 5 (gap = 3). Then shift both to
    # positions 7 and 10 (gap still = 3). Scores should match.
    q = np.random.randn(d_head)
    k = np.random.randn(d_head)

    def score_at(m, n):
        """q at position m, k at position n → attention score"""
        q_rot = apply_rope(q[None, :].repeat(max(m, n) + 1, 0), cos, sin)[m]
        k_rot = apply_rope(k[None, :].repeat(max(m, n) + 1, 0), cos, sin)[n]
        return q_rot @ k_rot

    s_2_5 = score_at(2, 5)    # gap 3
    s_7_10 = score_at(7, 10)  # gap 3 (shifted by 5)
    s_2_9 = score_at(2, 9)    # gap 7 (different!)
    print("\n2) Score depends only on relative distance:")
    print(f"   q@2,  k@5   (gap 3): {s_2_5:.6f}")
    print(f"   q@7,  k@10  (gap 3): {s_7_10:.6f}   <- identical")
    print(f"   q@2,  k@9   (gap 7): {s_2_9:.6f}   <- different")
    assert np.isclose(s_2_5, s_7_10), "RoPE relative-position property failed!"

    # --- Property 3: position 0 is unrotated ------------------
    print("\n3) Position 0 is rotated by 0 degrees (unchanged):")
    print(f"   max |v[0] - RoPE(v)[0]| = {np.abs(v[0] - v_rot[0]).max():.2e}")

    print("\nAll RoPE checks passed. Modern models apply this to Q and K")
    print("inside every attention layer instead of adding positions to")
    print("the embeddings (Block 2 style).")
