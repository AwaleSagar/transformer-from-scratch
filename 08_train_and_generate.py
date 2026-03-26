"""
============================================================
Block 8: Training Loop + Text Generation
============================================================
The grand finale! We train our transformer on the small corpus
and then use it to generate text.

Training (Supervised Learning):
  1. Feed a sequence of words into the model
  2. The model predicts the next word at each position
  3. Compare predictions with actual next words (cross-entropy loss)
  4. Compute gradients (backward pass)
  5. Update weights: param -= learning_rate × gradient  (SGD)
  6. Repeat until the model learns the patterns

Text Generation (Inference):
  1. Start with a prompt (e.g., "the cat")
  2. Feed it through the model → get probability for next word
  3. Pick a word (greedy = highest prob, or sample with temperature)
  4. Append the word to the prompt
  5. Repeat to generate a full sentence

What to expect:
  - Loss should drop from ~ln(vocab_size) ≈ 3.3 to below 1.0
  - The model will essentially memorize the tiny corpus
  - Generated text should resemble the training sentences
  - This is expected and correct for a ~15K parameter model
    trained on ~150 words!
============================================================
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_module import load
_b1 = load("01_tokenizer.py")
_b3 = load("03_attention.py")
_b7 = load("07_transformer.py")

Tokenizer = _b1.Tokenizer
CORPUS = _b1.CORPUS
create_training_data = _b1.create_training_data
softmax = _b3.softmax
Transformer = _b7.Transformer
cross_entropy_loss = _b7.cross_entropy_loss


# ─── SGD Optimizer ──────────────────────────────────────────

class SGD:
    """
    Stochastic Gradient Descent — the simplest optimizer.

    Update rule:
      param = param - learning_rate × gradient

    That's it! Each parameter moves in the direction that
    reduces the loss, scaled by the learning rate.

    Learning rate:
      - Too high → loss explodes (overshooting)
      - Too low  → learning is very slow
      - Just right → steady decrease in loss
    """

    def __init__(self, parameters_fn, lr=0.01):
        """
        Args:
            parameters_fn: A callable that returns list of (param, grad) tuples.
                           We use a function (not a list) because gradients are
                           recomputed each step.
            lr: Learning rate (step size).
        """
        self.parameters_fn = parameters_fn
        self.lr = lr

    def step(self):
        """
        Perform one optimization step.

        For each parameter that has a gradient, update:
          param -= lr * grad
        """
        for param, grad in self.parameters_fn():
            if grad is not None:
                param -= self.lr * grad

    def zero_grad(self):
        """
        Reset all gradients to zero.

        Must be called before each new backward pass, otherwise
        gradients accumulate across steps.
        """
        for param, grad in self.parameters_fn():
            if grad is not None:
                grad.fill(0)


# ─── Text Generation ───────────────────────────────────────

def generate(model, tokenizer, prompt, max_tokens=20, temperature=1.0):
    """
    Generate text autoregressively.

    At each step:
      1. Encode the current text as token IDs
      2. Run forward pass → logits for next word
      3. Apply temperature scaling to logits
      4. Sample from the probability distribution
      5. Append sampled token and repeat

    Temperature controls randomness:
      - temperature → 0: always pick highest-probability word (greedy)
      - temperature = 1: sample according to model's probabilities
      - temperature > 1: more random/creative

    Args:
        model: Trained Transformer model
        tokenizer: Tokenizer instance
        prompt: Starting text (string)
        max_tokens: How many words to generate
        temperature: Sampling temperature

    Returns:
        Generated text string (including prompt).
    """
    # Encode the prompt
    token_ids = tokenizer.encode(prompt)

    for _ in range(max_tokens):
        # Use the last max_len tokens (to stay within sequence length)
        context = np.array(token_ids[-model.max_len:])

        # Forward pass
        logits = model.forward(context)

        # Get logits for the last position (next word prediction)
        next_logits = logits[-1]                           # (vocab_size,)

        # Apply temperature
        if temperature < 0.01:
            # Greedy: pick the highest probability
            next_id = int(np.argmax(next_logits))
        else:
            # Temperature scaling
            scaled = next_logits / temperature
            probs = softmax(scaled)                        # (vocab_size,)

            # Sample from the distribution
            next_id = int(np.random.choice(len(probs), p=probs))

        token_ids.append(next_id)

    return tokenizer.decode(token_ids)


# ─── Training Loop ──────────────────────────────────────────

def train(model, training_pairs, tokenizer, epochs=100, lr=0.05,
          print_every=10, seed=42):
    """
    Train the transformer on next-word prediction.

    Args:
        model: Transformer model
        training_pairs: List of (input_ids, target_ids) from create_training_data
        tokenizer: For decoding during progress prints
        epochs: Number of full passes through the data
        lr: Learning rate
        print_every: Print loss every N epochs
        seed: Random seed for reproducibility

    Returns:
        List of (epoch, loss) for plotting.
    """
    rng = np.random.default_rng(seed)
    optimizer = SGD(model.parameters, lr=lr)
    loss_history = []

    print(f"\n{'='*55}")
    print(f"  Training")
    print(f"{'='*55}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Training pairs: {len(training_pairs)}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"{'='*55}\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Shuffle training data each epoch
        indices = rng.permutation(len(training_pairs))
        epoch_loss = 0.0

        for idx in indices:
            input_ids, target_ids = training_pairs[idx]
            input_arr = np.array(input_ids)
            target_arr = np.array(target_ids)

            # Forward pass
            logits = model.forward(input_arr)              # (seq_len, vocab_size)

            # Compute loss
            loss, grad_logits = cross_entropy_loss(logits, target_arr)
            epoch_loss += loss

            # Backward pass
            optimizer.zero_grad()
            model.backward(grad_logits)

            # Update parameters
            optimizer.step()

        # Average loss for this epoch
        avg_loss = epoch_loss / len(training_pairs)
        loss_history.append((epoch, avg_loss))

        # Print progress
        if epoch % print_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            # Generate a sample to show progress
            sample = generate(model, tokenizer, "the cat",
                              max_tokens=8, temperature=0.5)
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"Loss: {avg_loss:.4f}  "
                  f"Time: {elapsed:.1f}s  "
                  f"Sample: \"{sample}\"")

    total_time = time.time() - start_time
    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Final loss: {loss_history[-1][1]:.4f}")

    return loss_history


def print_loss_chart(loss_history, width=50):
    """
    Print an ASCII loss curve.

    Args:
        loss_history: List of (epoch, loss) tuples.
        width: Chart width in characters.
    """
    if not loss_history:
        return

    losses = [l for _, l in loss_history]
    max_loss = max(losses)
    min_loss = min(losses)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1.0

    print(f"\n  Loss Curve:")
    print(f"  {'─' * (width + 10)}")

    # Sample ~20 points for the chart, always including the last point
    step = max(1, len(loss_history) // 20)
    sampled = sorted(set(range(0, len(loss_history), step)) | {len(loss_history) - 1})
    for i in sampled:
        epoch, loss = loss_history[i]
        bar_len = int((loss - min_loss) / loss_range * width)
        bar = "█" * bar_len
        print(f"  E{epoch:4d} │{bar} {loss:.3f}")

    print(f"  {'─' * (width + 10)}")
    print(f"  Start: {losses[0]:.4f} → End: {losses[-1]:.4f}")
    print(f"  Reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")


# ─── Main: Train and Generate ──────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Block 8: Training & Text Generation")
    print("=" * 55)

    np.random.seed(42)

    # ── Step 1: Prepare data ──
    print("\n--- Data Preparation ---")
    tok = Tokenizer()
    tok.fit(CORPUS)
    all_ids = tok.encode(CORPUS)

    seq_len = 16
    training_pairs = create_training_data(all_ids, seq_len)
    print(f"  Vocabulary: {tok.vocab_size} words")
    print(f"  Corpus: {len(all_ids)} tokens")
    print(f"  Sequence length: {seq_len}")
    print(f"  Training pairs: {len(training_pairs)}")

    # ── Step 2: Build model ──
    print("\n--- Model ---")
    model = Transformer(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_heads=2,
        d_ff=128,
        n_layers=1,
        max_len=seq_len,
        seed=42,
    )
    print(f"  Architecture: Decoder-only Transformer")
    print(f"  Total parameters: {model.count_parameters():,}")

    # ── Step 3: Train ──
    loss_history = train(
        model=model,
        training_pairs=training_pairs,
        tokenizer=tok,
        epochs=100,
        lr=0.05,
        print_every=10,
        seed=42,
    )

    # ── Step 4: Visualize training ──
    print_loss_chart(loss_history)

    # ── Step 5: Generate text ──
    print(f"\n{'='*55}")
    print(f"  Text Generation")
    print(f"{'='*55}")

    prompts = [
        "the cat",
        "the dog",
        "a bird",
        "the big",
        "a small",
    ]

    print(f"\n--- Greedy Generation (temperature ≈ 0) ---")
    for prompt in prompts:
        text = generate(model, tok, prompt, max_tokens=12, temperature=0.01)
        print(f'  "{prompt}" → "{text}"')

    print(f"\n--- Sampled Generation (temperature = 0.5) ---")
    for prompt in prompts:
        text = generate(model, tok, prompt, max_tokens=12, temperature=0.5)
        print(f'  "{prompt}" → "{text}"')

    print(f"\n--- Creative Generation (temperature = 1.0) ---")
    for prompt in prompts:
        text = generate(model, tok, prompt, max_tokens=12, temperature=1.0)
        print(f'  "{prompt}" → "{text}"')

    # ── Step 6: Interactive (if desired) ──
    print(f"\n{'='*55}")
    print(f"  Congratulations! You built a transformer from scratch.")
    print(f"{'='*55}")
    print(f"\n  The model learned to predict next words from this vocabulary:")
    print(f"  {', '.join(w for w in tok.word_to_id if w not in ('<PAD>', '<UNK>'))}")
    print(f"\n  Try modifying CORPUS in 01_tokenizer.py to teach it")
    print(f"  different patterns, or increase n_layers/d_model for")
    print(f"  more capacity (at the cost of training time).")
