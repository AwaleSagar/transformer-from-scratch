"""
============================================================
Block 2: Token Embeddings + Positional Encoding
============================================================
Neural networks work with continuous vectors, not discrete
integer IDs. An embedding layer converts each token ID into
a dense vector of size d_model. We also add positional
information so the model knows *where* each token sits in
the sequence (since attention has no built-in notion of order).

Key concepts:
  - Token Embedding:  lookup table (vocab_size × d_model)
  - Positional Encoding: sine/cosine waves encode position
  - The two are summed to produce the final input to the
    transformer

From "Attention Is All You Need" (Vaswani et al., 2017):
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
============================================================
"""

import numpy as np
import sys
import os

# Allow importing from the same directory when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_module import load
_b1 = load("01_tokenizer.py")
Tokenizer, CORPUS = _b1.Tokenizer, _b1.CORPUS


# ─── Xavier (Glorot) Initialization ────────────────────────
def xavier_init(shape, seed=None):
    """
    Xavier uniform initialization.

    Keeps the variance of activations roughly equal across layers,
    which helps training converge.

    scale = sqrt(6 / (fan_in + fan_out))
    weights ~ Uniform(-scale, +scale)
    """
    rng = np.random.default_rng(seed)
    fan_in, fan_out = shape[0], shape[1]
    scale = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-scale, scale, size=shape)


class TokenEmbedding:
    """
    Learnable lookup table that maps token IDs → dense vectors.

    Forward:
      Given token_ids [t0, t1, ..., tN], return:
        embeddings[t0], embeddings[t1], ..., embeddings[tN]
      Shape: (seq_len,) → (seq_len, d_model)

    Backward:
      The gradient for an embedding row is the sum of all
      gradients at positions where that token appeared.
      (Only the rows that were looked up get updated.)
    """

    def __init__(self, vocab_size, d_model, seed=42):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.W = xavier_init((vocab_size, d_model), seed=seed)

        # Gradient accumulator (set during backward)
        self.grad_W = None

        # Cache for backward pass
        self._token_ids = None

    def forward(self, token_ids):
        """
        Args:
            token_ids: numpy array of shape (seq_len,), dtype int

        Returns:
            numpy array of shape (seq_len, d_model)
        """
        self._token_ids = token_ids  # save for backward
        return self.W[token_ids]     # fancy indexing = lookup

    def backward(self, grad_output):
        """
        Args:
            grad_output: numpy array of shape (seq_len, d_model)
                         — gradient of loss w.r.t. embedding output

        Returns:
            grad_W: gradient for the embedding table
                    (same shape as self.W, mostly zeros)
        """
        self.grad_W = np.zeros_like(self.W)
        # Accumulate gradients for each token that was looked up.
        # If a token appears multiple times, its gradients add up.
        np.add.at(self.grad_W, self._token_ids, grad_output)
        return self.grad_W

    def parameters(self):
        """Return list of (param_array, grad_array) for optimizer."""
        return [(self.W, self.grad_W)]


class PositionalEncoding:
    """
    Fixed (non-learnable) sinusoidal positional encoding.

    Each position gets a unique pattern of sine and cosine waves
    at different frequencies. This lets the model distinguish
    position 0 from position 5 from position 15, etc.

    The encoding is pre-computed for all positions up to max_len.

    Why sine/cosine?
      - It's bounded (values in [-1, 1])
      - Each position has a unique encoding
      - The model can learn to attend to relative positions
        because PE(pos+k) can be expressed as a linear function
        of PE(pos)
    """

    def __init__(self, d_model, max_len=512):
        self.d_model = d_model
        self.encoding = self._build_encoding(max_len, d_model)

    def _build_encoding(self, max_len, d_model):
        """
        Build the full positional encoding table.

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = np.zeros((max_len, d_model))

        # pos = [[0], [1], [2], ..., [max_len-1]]
        pos = np.arange(max_len).reshape(-1, 1)

        # Compute the division term: 10000^(2i/d_model)
        # Using log-space for numerical stability:
        #   10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        i = np.arange(0, d_model, 2)  # even indices: 0, 2, 4, ...
        div_term = np.exp(i * (-np.log(10000.0) / d_model))

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = np.sin(pos * div_term)  # even columns
        pe[:, 1::2] = np.cos(pos * div_term)  # odd columns

        return pe

    def forward(self, seq_len):
        """
        Args:
            seq_len (int): Length of the input sequence.

        Returns:
            numpy array of shape (seq_len, d_model)
        """
        return self.encoding[:seq_len]

    # No backward needed — positional encodings are fixed (no learnable params)


class Embedding:
    """
    Combined token + positional embedding.

    output = TokenEmbedding(token_ids) + PositionalEncoding(seq_len)

    This is the very first layer of the transformer.
    """

    def __init__(self, vocab_size, d_model, max_len=512, seed=42):
        self.token_emb = TokenEmbedding(vocab_size, d_model, seed=seed)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.d_model = d_model

    def forward(self, token_ids):
        """
        Args:
            token_ids: numpy array of shape (seq_len,), dtype int

        Returns:
            numpy array of shape (seq_len, d_model)
        """
        seq_len = len(token_ids)
        tok_emb = self.token_emb.forward(token_ids)   # (seq_len, d_model)
        pos_enc = self.pos_enc.forward(seq_len)        # (seq_len, d_model)
        return tok_emb + pos_enc

    def backward(self, grad_output):
        """
        Positional encoding is fixed, so all gradient goes
        to the token embedding.

        Args:
            grad_output: shape (seq_len, d_model)
        """
        return self.token_emb.backward(grad_output)

    def parameters(self):
        """Learnable parameters (only token embeddings)."""
        return self.token_emb.parameters()


# ─── Self-test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Block 2: Embeddings + Positional Encoding")
    print("=" * 55)

    # Setup
    d_model = 32
    tok = Tokenizer()
    tok.fit(CORPUS)
    print(f"\nVocab size: {tok.vocab_size}, d_model: {d_model}")

    # 1. Token Embedding
    print("\n--- Token Embedding ---")
    token_emb = TokenEmbedding(tok.vocab_size, d_model, seed=42)
    sample_ids = np.array(tok.encode("the cat sat on the mat"))
    emb_out = token_emb.forward(sample_ids)
    print(f"  Input IDs  : {sample_ids}")
    print(f"  Output shape: {emb_out.shape}")
    print(f"  First token vector (first 8 dims):")
    print(f"    {emb_out[0, :8].round(3)}")

    # 2. Positional Encoding
    print("\n--- Positional Encoding ---")
    pos_enc = PositionalEncoding(d_model)
    pe = pos_enc.forward(8)
    print(f"  Shape for 8 positions: {pe.shape}")
    print(f"\n  Encoding grid (8 positions × first 8 dims):")
    print(f"  {'pos':>4}", end="")
    for j in range(8):
        print(f" {'d'+str(j):>7}", end="")
    print()
    for i in range(8):
        print(f"  {i:4d}", end="")
        for j in range(8):
            print(f" {pe[i, j]:7.3f}", end="")
        print()

    # 3. Combined Embedding
    print("\n--- Combined Embedding ---")
    emb = Embedding(tok.vocab_size, d_model, seed=42)
    combined = emb.forward(sample_ids)
    print(f"  Input : '{tok.decode(sample_ids.tolist())}'")
    print(f"  IDs   : {sample_ids}")
    print(f"  Output shape: {combined.shape}")

    # Show that different positions of the same word get different vectors
    # "the" appears at positions 0 and 4
    print(f"\n  'the' at position 0 (first 8 dims): {combined[0, :8].round(3)}")
    print(f"  'the' at position 4 (first 8 dims): {combined[4, :8].round(3)}")
    print(f"  → Different! Positional encoding makes each position unique.")

    # 4. Backward pass test
    print("\n--- Backward Pass ---")
    fake_grad = np.ones_like(combined) * 0.1
    emb.backward(fake_grad)
    print(f"  Gradient propagated through embedding ✓")
