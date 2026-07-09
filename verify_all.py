"""
============================================================
verify_all.py — the repo's verification gate
============================================================
One numpy-only harness (no pytest, no new deps) that IMPORTS
each block and re-checks its core guarantee with real asserts,
so a broken gradient or forward pass fails the build instead of
just printing "✗ Mismatch!".

What it checks (see the tech plan's verification matrix):

  Blocks 3/4/5 : numerical vs analytical gradient (the real
                 gate — these used to be print-only)
  Blocks 6/7   : backward runs and every parameter gets a
                 finite (non-NaN) gradient — smoke test only
  Blocks 9-12  : forward-pass invariants (RoPE relative
                 property, RMSNorm/SwiGLU shapes, KV-cache
                 equivalence, logits shape)
  Block 13     : each token is routed to exactly top_k experts

Usage:
    python verify_all.py     # exit 0 if all pass, non-zero on any failure

A failed assert raises (traceback + non-zero exit); nothing is
caught or swallowed here. The summary line prints only if every
check passed.
============================================================
"""

import numpy as np
from load_module import load

checks = 0


def ok(msg):
    """Record a passed check and print a one-line confirmation."""
    global checks
    checks += 1
    print(f"  ✓ {msg}")


# ── Blocks 3/4/5: numerical vs analytical gradient (the real gate) ──

def check_block3():
    b3 = load("03_attention.py")
    np.random.seed(42)
    seq_len, d_k = 4, 8
    Q, K, V = (np.random.randn(seq_len, d_k) for _ in range(3))
    out, cache = b3.attention_forward(Q, K, V)
    fake = np.ones_like(out) * 0.1
    grad_Q, _, _ = b3.attention_backward(fake, cache)
    eps = 1e-5
    Qp, Qm = Q.copy(), Q.copy()
    Qp[0, 0] += eps
    Qm[0, 0] -= eps
    op, _ = b3.attention_forward(Qp, K, V)
    om, _ = b3.attention_forward(Qm, K, V)
    numerical = np.sum(fake * (op - om)) / (2 * eps)
    assert abs(numerical - grad_Q[0, 0]) < 1e-5, "Block 3 grad check failed"
    ok("Block 3: attention grad_Q matches numerical (< 1e-5)")


def check_block4():
    b4 = load("04_multi_head_attention.py")
    np.random.seed(42)
    d_model, n_heads, seq_len = 32, 2, 6
    mha = b4.MultiHeadAttention(d_model, n_heads, seed=42)
    mask = b4.create_causal_mask(seq_len)
    X = np.random.randn(seq_len, d_model)
    fake = np.random.randn(seq_len, d_model) * 0.01
    mha.forward(X, mask=mask)
    mha.backward(fake)
    eps = 1e-5
    loss = lambda o: np.sum(fake * o)
    orig = mha.W_Q[0, 0]
    mha.W_Q[0, 0] = orig + eps
    op = mha.forward(X, mask=mask)
    mha.W_Q[0, 0] = orig - eps
    om = mha.forward(X, mask=mask)
    mha.W_Q[0, 0] = orig
    numerical = (loss(op) - loss(om)) / (2 * eps)
    assert abs(numerical - mha.grad_W_Q[0, 0]) < 1e-4, "Block 4 grad check failed"
    ok("Block 4: MHA grad_W_Q[0,0] matches numerical (< 1e-4)")


def check_block5():
    b5 = load("05_feedforward_layernorm.py")
    np.random.seed(42)
    d_model, seq_len = 32, 6
    ln = b5.LayerNorm(d_model)
    X = np.random.randn(seq_len, d_model)
    fake = np.random.randn(seq_len, d_model) * 0.01
    ln.forward(X)
    ln.backward(fake)
    eps = 1e-5
    loss = lambda o: np.sum(fake * o)
    orig = ln.gamma[0]
    ln.gamma[0] = orig + eps
    op = ln.forward(X)
    ln.gamma[0] = orig - eps
    om = ln.forward(X)
    ln.gamma[0] = orig
    numerical = (loss(op) - loss(om)) / (2 * eps)
    assert abs(numerical - ln.grad_gamma[0]) < 1e-4, "Block 5 grad check failed"
    ok("Block 5: LayerNorm grad_gamma[0] matches numerical (< 1e-4)")


# ── Blocks 6/7: backward runs, every param gets a finite gradient (smoke) ──

def _assert_finite_grads(params, label):
    assert len(params) > 0, f"{label}: no parameters"
    for i, (_, g) in enumerate(params):
        assert g is not None, f"{label}: param {i} has no gradient"
        assert np.all(np.isfinite(g)), f"{label}: param {i} gradient not finite"


def check_block6():
    b6 = load("06_transformer_block.py")
    np.random.seed(42)
    d_model, n_heads, d_ff, seq_len = 32, 2, 128, 8
    block = b6.TransformerBlock(d_model, n_heads, d_ff, seed=42)
    mask = b6.create_causal_mask(seq_len)
    X = np.random.randn(seq_len, d_model)
    block.forward(X, mask=mask)
    block.backward(np.random.randn(seq_len, d_model) * 0.01)
    _assert_finite_grads(block.parameters(), "Block 6")
    ok("Block 6: backward runs, all params have finite gradients")


def check_block7():
    b7 = load("07_transformer.py")
    tok = b7.Tokenizer()
    tok.fit(b7.CORPUS)
    model = b7.Transformer(vocab_size=tok.vocab_size, seed=42)
    ids = np.array(tok.encode("the cat sat on the mat and the dog")[:8])
    logits = model.forward(ids)
    _, grad = b7.cross_entropy_loss(logits[:-1], ids[1:])
    full = np.zeros_like(logits)
    full[:-1] = grad
    model.backward(full)
    _assert_finite_grads(model.parameters(), "Block 7")
    ok("Block 7: full backward runs, all params have finite gradients")


# ── Blocks 9-12: forward-pass invariants ──

def check_block9():
    b9 = load("09_rope.py")
    d_head = 8
    cos, sin = b9.rope_frequencies(d_head, max_seq_len=64)
    np.random.seed(0)
    q, k = np.random.randn(d_head), np.random.randn(d_head)

    def score_at(m, n):
        qr = b9.apply_rope(q[None, :].repeat(max(m, n) + 1, 0), cos, sin)[m]
        kr = b9.apply_rope(k[None, :].repeat(max(m, n) + 1, 0), cos, sin)[n]
        return qr @ kr

    # Same relative gap (3) at different absolute positions → same score.
    assert np.isclose(score_at(2, 5), score_at(7, 10)), "Block 9 RoPE relative property failed"
    ok("Block 9: RoPE score depends only on relative position")


def check_block10():
    b10 = load("10_rmsnorm_swiglu.py")
    np.random.seed(0)
    d_model, d_ff, seq_len = 32, int(8 * 32 / 3), 6
    x = np.random.randn(seq_len, d_model) * 3 + 1.5
    y = b10.RMSNorm(d_model).forward(x)
    rms = np.sqrt(np.mean(y ** 2, axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-3), "Block 10 RMSNorm did not normalize RMS to ~1"
    out = b10.SwiGLU(d_model, d_ff).forward(x)
    assert out.shape == x.shape, "Block 10 SwiGLU changed shape"
    ok("Block 10: RMSNorm rms≈1 and SwiGLU preserves shape")


def check_block11():
    b11 = load("11_gqa_kv_cache.py")
    np.random.seed(1)
    d_model, n_q, n_kv, seq_len = 32, 4, 2, 10
    x = np.random.randn(seq_len, d_model)
    attn = b11.GroupedQueryAttention(d_model, n_q, n_kv)
    full = attn.forward(x)
    attn.reset_cache()
    cached = np.concatenate([attn.forward(x[t:t + 1], use_cache=True)
                             for t in range(seq_len)], axis=0)
    err = np.abs(full - cached).max()
    assert err < 1e-9, f"Block 11 KV cache mismatch (err={err:.2e})"
    ok("Block 11: KV-cached generation matches full forward (err < 1e-9)")


def check_block12():
    b12 = load("12_modern_transformer.py")
    np.random.seed(7)
    vocab_size, seq_len = 24, 10
    model = b12.ModernTransformer(vocab_size)
    ids = list(np.random.randint(0, vocab_size, seq_len))
    logits = model.forward(ids)
    assert logits.shape == (seq_len, vocab_size), "Block 12 logits shape wrong"
    ok("Block 12: forward produces (seq_len, vocab_size) logits")


# ── Block 13: top-k routing invariant ──

def check_block13():
    b13 = load("13_mixture_of_experts.py")
    np.random.seed(0)
    d_model, d_ff, seq_len = 32, int(8 * 32 / 3), 12
    for top_k in (1, 2):
        moe = b13.MoELayer(d_model, d_ff, n_experts=4, top_k=top_k, seed=0)
        out = moe.forward(np.random.randn(seq_len, d_model))
        assert out.shape == (seq_len, d_model), "Block 13 output shape wrong"
        active = moe.last_mask.sum(axis=1)
        assert np.all(active == top_k), \
            f"Block 13: tokens not routed to exactly top_k={top_k} experts"
    ok("Block 13: each token routed to exactly top_k experts (k=1,2)")


def main():
    print("=" * 60)
    print("verify_all.py — importing blocks and asserting invariants")
    print("=" * 60)
    for check in (check_block3, check_block4, check_block5,
                  check_block6, check_block7,
                  check_block9, check_block10, check_block11, check_block12,
                  check_block13):
        check()
    print("=" * 60)
    print(f"All {checks} checks passed ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
