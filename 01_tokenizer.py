"""
============================================================
Block 1: Tokenizer — Vocabulary & Tokenization
============================================================
A tokenizer converts raw text into a sequence of integers
(token IDs) that a neural network can process, and converts
model output back into readable text.

Key concepts:
  - Vocabulary: the set of all unique words the model knows
  - Encoding: text → list of integer IDs
  - Decoding: list of integer IDs → text
  - Special tokens: <PAD> (padding) and <UNK> (unknown words)

We also prepare training data here: sliding windows of
token IDs for next-word prediction.
============================================================
"""

import re


class Tokenizer:
    """
    A simple word-level tokenizer.

    Usage:
        tok = Tokenizer()
        tok.fit("the cat sat on the mat")
        ids = tok.encode("the cat")    # → [2, 3]
        text = tok.decode(ids)         # → "the cat"
    """

    def __init__(self):
        # Special tokens get the first two IDs
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        # Mappings (populated by fit())
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

    def fit(self, text):
        """
        Build vocabulary from a text corpus.

        Steps:
          1. Lowercase and split into words
          2. Collect unique words
          3. Assign an integer ID to each word

        Args:
            text (str): The training corpus as a single string.
        """
        # Step 1: Tokenize — lowercase, split on non-alphanumeric chars
        words = self._tokenize(text)

        # Step 2: Get unique words in order of first appearance
        unique_words = []
        seen = set()
        for w in words:
            if w not in seen:
                unique_words.append(w)
                seen.add(w)

        # Step 3: Build mappings — special tokens first
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
        }
        for i, word in enumerate(unique_words):
            self.word_to_id[word] = i + 2  # offset by 2 for special tokens

        # Reverse mapping: ID → word
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)

    def encode(self, text):
        """
        Convert text to a list of token IDs.

        Unknown words (not in vocabulary) map to <UNK> (ID=1).

        Args:
            text (str): Input text string.

        Returns:
            list[int]: Token IDs.
        """
        words = self._tokenize(text)
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(w, unk_id) for w in words]

    def decode(self, token_ids):
        """
        Convert a list of token IDs back to text.

        Args:
            token_ids (list[int]): Sequence of token IDs.

        Returns:
            str: Decoded text string.
        """
        words = [self.id_to_word.get(idx, self.unk_token) for idx in token_ids]
        return " ".join(words)

    def _tokenize(self, text):
        """
        Split text into lowercase word tokens.

        Uses regex to extract sequences of letters/digits,
        discarding punctuation and whitespace.
        """
        return re.findall(r"[a-z0-9]+", text.lower())


def create_training_data(token_ids, seq_len):
    """
    Create input/target pairs for next-word prediction.

    Given a stream of token IDs, we slide a window of size
    (seq_len + 1) across it. For each window:
      - input  = first seq_len tokens
      - target = last  seq_len tokens (shifted by 1)

    Example (seq_len=4):
        token_ids = [2, 3, 4, 5, 6, 7, 8]

        Window 1: [2, 3, 4, 5, 6]
          input:  [2, 3, 4, 5]
          target: [3, 4, 5, 6]

        Window 2: [3, 4, 5, 6, 7]
          input:  [3, 4, 5, 6]
          target: [4, 5, 6, 7]

    Args:
        token_ids (list[int]): Full corpus as token IDs.
        seq_len (int): Context window size.

    Returns:
        list of (input_ids, target_ids) tuples.
    """
    pairs = []
    for i in range(len(token_ids) - seq_len):
        input_ids = token_ids[i : i + seq_len]
        target_ids = token_ids[i + 1 : i + seq_len + 1]
        pairs.append((input_ids, target_ids))
    return pairs


# ─── Built-in training corpus ───────────────────────────────
# A small, repetitive text so the model can memorize patterns.
# Repetition is intentional — it helps a tiny model learn.

CORPUS = """
the cat sat on the mat and the dog sat on the rug
the cat chased the dog around the big garden
the dog chased the cat around the small house
a bird sat on the fence and watched the cat
the cat watched the bird and the dog watched the cat
the bird flew over the garden and the house
the dog sat on the mat and watched the bird
a small cat sat on a big mat near the garden
the big dog chased a small bird around the house
the cat and the dog sat on the mat together
the bird sat on the fence near the small garden
a big cat watched a small dog near the house
the dog and the cat chased the bird around the garden
the bird flew over the mat and the rug and the fence
the small cat and the big dog sat on the rug together
"""


# ─── Self-test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Block 1: Tokenizer")
    print("=" * 55)

    # 1. Build vocabulary
    tok = Tokenizer()
    tok.fit(CORPUS)

    print(f"\nVocabulary size: {tok.vocab_size}")
    print(f"\nVocabulary (word → ID):")
    for word, idx in sorted(tok.word_to_id.items(), key=lambda x: x[1]):
        print(f"  {idx:3d} → '{word}'")

    # 2. Encode / Decode
    sample = "the cat sat on the mat"
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)

    print(f"\nEncode/Decode test:")
    print(f"  Original : '{sample}'")
    print(f"  Encoded  : {encoded}")
    print(f"  Decoded  : '{decoded}'")

    # 3. Unknown word handling
    unk_test = "the elephant sat on the moon"
    unk_encoded = tok.encode(unk_test)
    print(f"\nUnknown word test:")
    print(f"  Input   : '{unk_test}'")
    print(f"  Encoded : {unk_encoded}")
    print(f"  Decoded : '{tok.decode(unk_encoded)}'")

    # 4. Training data
    all_ids = tok.encode(CORPUS)
    seq_len = 16
    pairs = create_training_data(all_ids, seq_len)

    print(f"\nTraining data:")
    print(f"  Corpus length : {len(all_ids)} tokens")
    print(f"  Sequence length: {seq_len}")
    print(f"  Training pairs : {len(pairs)}")
    print(f"\n  First 3 pairs:")
    for i, (inp, tgt) in enumerate(pairs[:3]):
        print(f"    Pair {i+1}:")
        print(f"      Input : {inp}")
        print(f"              {tok.decode(inp)}")
        print(f"      Target: {tgt}")
        print(f"              {tok.decode(tgt)}")
