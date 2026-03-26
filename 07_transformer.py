"""
============================================================
Block 7: Full Decoder-Only Transformer
============================================================
We now stack all previous blocks into a complete transformer
language model. This is a decoder-only architecture (like GPT):

  Token IDs
      │
      ▼
  ┌──────────────────────────┐
  │  Token Embedding         │  ← learnable lookup table
  │  + Positional Encoding   │  ← fixed sine/cosine
  └──────────────────────────┘
      │
      ▼
  ┌──────────────────────────┐
  │  Transformer Block × N   │  ← N stacked decoder blocks
  │  (Attention + FFN)       │     (N=1 for our tiny model)
  └──────────────────────────┘
      │
      ▼
  ┌──────────────────────────┐
  │  Final LayerNorm         │  ← stabilize before output
  └──────────────────────────┘
      │
      ▼
  ┌──────────────────────────┐
  │  Output Linear           │  ← project to vocab_size
  │  (d_model → vocab_size)  │     to predict next word
  └──────────────────────────┘
      │
      ▼
    Logits (seq_len, vocab_size)
      │
      ▼
    Softmax → probabilities for each next word

We also implement cross-entropy loss here, which measures
how well the model's predictions match the actual next words.

Model Config (tiny, CPU-friendly):
  vocab_size ≈ 50      d_model = 32
  n_heads = 2          d_ff = 128
  n_layers = 1         seq_len = 16
  Total params ≈ 15K
============================================================
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_module import load
_b1 = load("01_tokenizer.py")
_b2 = load("02_embeddings.py")
_b3 = load("03_attention.py")
_b5 = load("05_feedforward_layernorm.py")
_b6 = load("06_transformer_block.py")

Tokenizer = _b1.Tokenizer
CORPUS = _b1.CORPUS
create_training_data = _b1.create_training_data
Embedding = _b2.Embedding
create_causal_mask = _b3.create_causal_mask
softmax = _b3.softmax
Linear = _b5.Linear
LayerNorm = _b5.LayerNorm
TransformerBlock = _b6.TransformerBlock


# ─── Cross-Entropy Loss ────────────────────────────────────

def cross_entropy_loss(logits, targets):
    """
    Compute cross-entropy loss for next-word prediction.

    For each position, the loss measures how surprised the model
    is by the actual next word. Lower loss = better predictions.

    Cross-entropy = -Σ log(P(correct word))

    We use the log-sum-exp trick for numerical stability:
      log(softmax(x_i)) = x_i - log(Σ exp(x_j))

    Args:
        logits:  Raw model output, shape (seq_len, vocab_size)
        targets: Ground truth token IDs, shape (seq_len,)

    Returns:
        loss: Scalar, average cross-entropy loss
        grad_logits: Gradient of loss w.r.t. logits,
                     shape (seq_len, vocab_size)
    """
    seq_len, vocab_size = logits.shape

    # Stable softmax
    probs = softmax(logits)                                # (seq_len, vocab_size)

    # Clamp probabilities to avoid log(0)
    probs_clamped = np.clip(probs, 1e-8, 1.0)

    # Loss: average negative log-probability of correct tokens
    # For each position i, we want -log(probs[i, targets[i]])
    correct_log_probs = -np.log(probs_clamped[np.arange(seq_len), targets])
    loss = np.mean(correct_log_probs)

    # Gradient of cross-entropy w.r.t. logits:
    #   grad = probs - one_hot(targets)
    #   (This elegant formula comes from combining softmax + cross-entropy)
    grad_logits = probs.copy()
    grad_logits[np.arange(seq_len), targets] -= 1.0
    grad_logits /= seq_len                                 # normalize by seq_len

    return loss, grad_logits


# ─── Transformer Model ─────────────────────────────────────

class Transformer:
    """
    Decoder-only Transformer language model.

    Predicts the next word at each position in the sequence.

    Architecture:
      Embedding → TransformerBlock × n_layers → LayerNorm → Linear → Logits
    """

    def __init__(self, vocab_size, d_model=32, n_heads=2,
                 d_ff=128, n_layers=1, max_len=512, seed=42):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.max_len = max_len

        # ── Build the model ──
        # 1. Embedding (token + positional)
        self.embedding = Embedding(vocab_size, d_model, max_len, seed=seed)

        # 2. Transformer blocks
        self.blocks = []
        for i in range(n_layers):
            block = TransformerBlock(
                d_model, n_heads, d_ff,
                seed=seed + i * 1000
            )
            self.blocks.append(block)

        # 3. Final layer norm
        self.final_norm = LayerNorm(d_model)

        # 4. Output projection: d_model → vocab_size
        self.output_proj = Linear(d_model, vocab_size, seed=seed + 9999)

        # Cache for causal masks, keyed by sequence length
        self._mask_cache = {}

    def forward(self, token_ids):
        """
        Full forward pass: token IDs → logits.

        Args:
            token_ids: numpy array of shape (seq_len,), dtype int

        Returns:
            logits: shape (seq_len, vocab_size)
                    — raw scores for each word at each position
        """
        seq_len = len(token_ids)

        # Get or create causal mask for this sequence length
        if seq_len not in self._mask_cache:
            self._mask_cache[seq_len] = create_causal_mask(seq_len)
        mask = self._mask_cache[seq_len]

        # 1. Embed tokens + add positional encoding
        x = self.embedding.forward(token_ids)              # (seq_len, d_model)

        # 2. Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask=mask)                # (seq_len, d_model)

        # 3. Final layer normalization
        x = self.final_norm.forward(x)                     # (seq_len, d_model)

        # 4. Project to vocabulary size
        logits = self.output_proj.forward(x)               # (seq_len, vocab_size)

        return logits

    def backward(self, grad_logits):
        """
        Full backward pass: gradient of loss w.r.t. logits → update all gradients.

        Args:
            grad_logits: shape (seq_len, vocab_size)
        """
        # 4. Output projection backward
        grad = self.output_proj.backward(grad_logits)      # (seq_len, d_model)

        # 3. Final layer norm backward
        grad = self.final_norm.backward(grad)              # (seq_len, d_model)

        # 2. Transformer blocks backward (reverse order!)
        for block in reversed(self.blocks):
            grad = block.backward(grad)                    # (seq_len, d_model)

        # 1. Embedding backward
        self.embedding.backward(grad)

    def parameters(self):
        """
        Collect ALL learnable parameters from the entire model.

        Returns:
            List of (parameter_array, gradient_array) tuples.
        """
        params = []

        # Embedding parameters
        params += self.embedding.parameters()

        # Transformer block parameters
        for block in self.blocks:
            params += block.parameters()

        # Final layer norm
        params += self.final_norm.parameters()

        # Output projection
        params += self.output_proj.parameters()

        return params

    def count_parameters(self):
        """Total number of learnable parameters."""
        return sum(p.size for p, _ in self.parameters())


# ─── Self-test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Block 7: Full Transformer Model")
    print("=" * 55)

    # Build tokenizer
    tok = Tokenizer()
    tok.fit(CORPUS)
    print(f"\nVocab size: {tok.vocab_size}")

    # Model config
    d_model = 32
    n_heads = 2
    d_ff = 128
    n_layers = 1
    seq_len = 16

    # Build model
    model = Transformer(
        vocab_size=tok.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        seed=42,
    )

    print(f"\n--- Model Architecture ---")
    print(f"  d_model:    {d_model}")
    print(f"  n_heads:    {n_heads}")
    print(f"  d_k:        {d_model // n_heads}")
    print(f"  d_ff:       {d_ff}")
    print(f"  n_layers:   {n_layers}")
    print(f"  vocab_size: {tok.vocab_size}")
    print(f"  max_len:    {model.max_len}")
    print(f"\n  Total parameters: {model.count_parameters():,}")

    # Forward pass
    print(f"\n--- Forward Pass ---")
    sample = "the cat sat on the mat and the dog sat on the rug the cat"
    token_ids = np.array(tok.encode(sample)[:seq_len])
    print(f"  Input: '{tok.decode(token_ids.tolist())}'")
    print(f"  Token IDs: {token_ids}")

    logits = model.forward(token_ids)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

    # Convert logits to probabilities, show top predictions
    probs = softmax(logits)
    print(f"\n  Top-3 predictions at each position:")
    for i in range(min(6, seq_len)):
        top3_idx = np.argsort(probs[i])[-3:][::-1]
        actual = token_ids[i+1] if i+1 < seq_len else -1
        preds = [(tok.id_to_word.get(idx, "?"), probs[i, idx]) for idx in top3_idx]
        actual_word = tok.id_to_word.get(actual, "—")
        print(f"    After '{tok.id_to_word[token_ids[i]]:>8s}' → "
              f"predict: {preds[0][0]}({preds[0][1]:.2f}) "
              f"{preds[1][0]}({preds[1][1]:.2f}) "
              f"{preds[2][0]}({preds[2][1]:.2f}) "
              f" | actual: {actual_word}")

    # Loss computation
    print(f"\n--- Loss ---")
    targets = token_ids[1:]   # shifted by 1 for next-word prediction
    loss_logits = logits[:-1] # all positions except last
    loss, grad = cross_entropy_loss(loss_logits, targets)
    print(f"  Loss: {loss:.4f}")
    print(f"  (Random baseline ≈ {np.log(tok.vocab_size):.4f} = ln(vocab_size))")
    print(f"  Grad shape: {grad.shape}")

    # Full backward pass
    print(f"\n--- Backward Pass ---")
    # Pad gradient to full sequence length (last position has no target)
    full_grad = np.zeros_like(logits)
    full_grad[:-1] = grad
    model.backward(full_grad)

    # Verify all parameters got gradients
    params = model.parameters()
    has_grad = sum(1 for _, g in params if g is not None)
    print(f"  Parameters with gradients: {has_grad}/{len(params)}")
    print(f"  Backward pass complete ✓")

    # Show gradient magnitudes
    print(f"\n--- Gradient Magnitudes ---")
    component_names = (
        ["Embedding"]
        + [f"Block {i} param {j}" for i in range(n_layers)
           for j in range(len(model.blocks[0].parameters()))]
        + ["Final LN γ", "Final LN β"]
        + ["Output W", "Output b"]
    )
    for name, (p, g) in zip(component_names, params):
        if g is not None:
            print(f"  {name:25s}: |grad| = {np.abs(g).mean():.6f}")
