"""
Byte-level Byte-Pair Encoding (BPE) tokenizer — from scratch.

This is the tokenizer GPT-2 / GPT-3 / GPT-4 use (with different vocab sizes and
merge tables). We build it the "minbpe" way (à la Karpathy) but keep the two
production-grade details that matter for real models:

1. **Byte-level, not char-level.** We operate on the raw UTF-8 *bytes* of the
   text, so the base vocabulary is exactly 256 symbols and *any* Unicode string
   round-trips losslessly — no `<UNK>`, ever. Emoji, Chinese, control
   characters: all just byte sequences.

2. **Regex pre-tokenization (GPT-2 pattern).** Before merging, we split text
   into chunks (words, numbers, punctuation runs) using GPT-2's regex. Merges
   are only ever learned *within* a chunk, never across a chunk boundary. This
   is why BPE tokens never span a space+word boundary in a weird way, and why
   " dog" and "dog" are different tokens.

Public API (mirrors the repo's SimpleTokenizer so it's a drop-in alternative):
    tok = BPETokenizer()
    tok.train(text, vocab_size=5000)
    ids   = tok.encode("hello world")
    text  = tok.decode(ids)          # decode(encode(s)) == s  for any UTF-8 s
    tok.save("tok.json"); BPETokenizer.load("tok.json")

The algorithm
-------------
Training repeatedly finds the most frequent adjacent pair of token ids across
the corpus and "merges" it into a brand-new id, recording the merge rule. We
start from the 256 byte values and add one new id per merge until we hit the
target vocab size. Encoding replays those merges in the order they were learned;
decoding just concatenates the byte strings each id expands to.

Why BPE at all? Char-level gives tiny vocab but very long sequences (attention
is O(n^2)); word-level gives short sequences but a huge, brittle vocab with an
OOV problem. BPE interpolates: frequent words become single tokens, rare words
fall back to sub-word pieces, and the byte base guarantees no OOV.
"""

from __future__ import annotations

import json
import regex as re
from collections import Counter
from typing import Dict, List, Tuple, Optional


# GPT-2's pre-tokenization pattern. It isolates:
#   's 't 're 've 'm 'll 'd  -> common English contractions (kept attached)
#   ?\p{L}+                  -> a run of letters, optionally with one leading space
#   ?\p{N}+                  -> a run of digits, optionally with one leading space
#   ?[^\s\p{L}\p{N}]+        -> a run of punctuation/symbols
#   \s+(?!\S) | \s+          -> trailing whitespace handling
# The leading " ?" is what makes " dog" tokenize as one chunk (space glued to
# the word) — a huge part of why GPT tokenizers "feel" the way they do.
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _get_stats(ids: List[int], counts: Optional[Counter] = None) -> Counter:
    """Count occurrences of each adjacent pair. counts is updated in place if given."""
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):  # consecutive elements
        counts[pair] += 1
    return counts


def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out: List[int] = []
    i = 0
    n = len(ids)
    while i < n:
        # If we're not at the last position and the pair matches here, merge it.
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """A byte-level BPE tokenizer with GPT-2-style regex pre-tokenization."""

    def __init__(self, pattern: Optional[str] = None):
        # merges: (int, int) -> int   the learned merge rules, in learned order
        self.merges: Dict[Tuple[int, int], int] = {}
        # vocab: int -> bytes         id -> the byte string it expands to
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # special_tokens: str -> int  reserved ids that bypass BPE (e.g. <|endoftext|>)
        self.special_tokens: Dict[str, int] = {}
        self._special_inv: Dict[int, str] = {}
        self.pattern = pattern or GPT2_SPLIT_PATTERN
        self._compiled = re.compile(self.pattern)

    # ------------------------------------------------------------------ train
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """
        Learn a BPE merge table from `text`.

        vocab_size must be >= 256 (the byte base). We perform `vocab_size - 256`
        merges. Pre-tokenization means we count pair stats across all chunks but
        only ever merge within a chunk (the chunk lists are kept separate).

        Performance: a naive trainer rescans every token on every merge — O(merges
        × corpus), which is minutes for a 5k vocab. Instead we collapse the corpus
        to **unique pre-tokens weighted by their frequency** (natural text repeats
        words heavily), and after each merge only re-scan the words that actually
        contained the merged pair. This is the standard BPE-trainer optimization
        (Sennrich 2016 / HuggingFace) and is exactly equivalent in output.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256 (the byte base)"
        num_merges = vocab_size - 256

        # 1. Pre-tokenize, then collapse to unique chunks -> frequency.
        chunk_counts = Counter(self._compiled.findall(text))
        # Each unique word: a mutable list of byte ids plus how often it occurs.
        words: List[List[int]] = [list(ch.encode("utf-8")) for ch in chunk_counts]
        freqs: List[int] = list(chunk_counts.values())

        # 2. Reset to the byte base and learn merges.
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Global pair stats and an index: pair -> set of word indices containing it.
        stats: Counter = Counter()
        pair_to_words: dict = {}
        for wi, ids in enumerate(words):
            f = freqs[wi]
            for pair in zip(ids, ids[1:]):
                stats[pair] += f
                pair_to_words.setdefault(pair, set()).add(wi)

        for step in range(num_merges):
            if not stats:
                break  # corpus fully merged (all words length 1)

            # Most frequent pair wins; ties broken by pair value for determinism.
            best_pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            new_id = 256 + step
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            occurrences = stats[best_pair]
            # Only touch words that actually contain best_pair. For each, subtract
            # its old adjacent-pair contributions, merge, then add the new ones —
            # keeping `stats` and the index incrementally correct.
            for wi in list(pair_to_words.get(best_pair, ())):
                ids = words[wi]
                f = freqs[wi]
                for pair in zip(ids, ids[1:]):
                    stats[pair] -= f
                    if stats[pair] <= 0:
                        del stats[pair]
                    s = pair_to_words.get(pair)
                    if s is not None:
                        s.discard(wi)
                merged = _merge(ids, best_pair, new_id)
                words[wi] = merged
                for pair in zip(merged, merged[1:]):
                    stats[pair] += f
                    pair_to_words.setdefault(pair, set()).add(wi)
            pair_to_words.pop(best_pair, None)

            if verbose:
                print(
                    f"merge {step + 1}/{num_merges}: {best_pair} -> {new_id} "
                    f"({self.vocab[new_id]!r}) had {occurrences} occurrences"
                )

    # --------------------------------------------------------------- specials
    def register_special_tokens(self, tokens: List[str]) -> None:
        """
        Reserve ids for special tokens (e.g. ['<|endoftext|>', '<|user|>']).

        Specials get ids at the top of the current vocab and are matched
        verbatim during encoding — they never participate in BPE merges. Used
        by M5 chat templates.
        """
        base = len(self.vocab)
        for offset, tok in enumerate(tokens):
            if tok in self.special_tokens:
                continue
            idx = base + offset
            self.special_tokens[tok] = idx
            self._special_inv[idx] = tok

    # ------------------------------------------------------------- encode/decode
    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        """BPE-encode a single pre-token's raw bytes by replaying learned merges."""
        ids = list(chunk_bytes)
        # Greedily apply the *lowest-numbered* applicable merge (i.e. the merge
        # that was learned earliest) until no learned pair remains. This exactly
        # reconstructs training-time merge order.
        while len(ids) >= 2:
            stats = _get_stats(ids)
            # Among pairs present, pick the one with the smallest merge id.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing left to merge
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> List[int]:
        """Encode text, ignoring special tokens (treat them as literal text)."""
        ids: List[int] = []
        for chunk in self._compiled.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special: str = "all") -> List[int]:
        """
        Encode text to token ids.

        If special tokens are registered and `allowed_special == "all"`, the
        text is split on any special-token string first so those map to their
        reserved id; everything between them is BPE-encoded normally.
        """
        if not self.special_tokens or allowed_special == "none":
            return self.encode_ordinary(text)

        # Split on the special tokens while keeping them (regex group capture).
        specials_re = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        parts = re.split(specials_re, text)
        ids: List[int] = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to a string (lossless for any encoded text)."""
        parts: List[bytes] = []
        for idx in ids:
            if idx in self.vocab:
                parts.append(self.vocab[idx])
            elif idx in self._special_inv:
                parts.append(self._special_inv[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        # errors="replace" only triggers if someone hands us corrupt ids; a
        # properly-encoded id stream always yields valid UTF-8.
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------- io
    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def save(self, path: str) -> None:
        """Serialize to JSON (merges + specials + pattern). No pickle, git-friendly."""
        data = {
            "pattern": self.pattern,
            # merges saved as ordered list of [a, b] so replay order is preserved.
            "merges": [[a, b] for (a, b) in self.merges.keys()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(pattern=data["pattern"])
        # Rebuild merges and vocab in learned order.
        for step, (a, b) in enumerate(data["merges"]):
            new_id = 256 + step
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        specials = data.get("special_tokens", {})
        # Preserve the exact ids that were saved.
        for stok, idx in specials.items():
            tok.special_tokens[stok] = idx
            tok._special_inv[idx] = stok
        return tok


if __name__ == "__main__":
    # Tiny self-demo: train on this file's own docstring and round-trip a string.
    sample = (
        "The quick brown fox jumps over the lazy dog. "
        "BPE merges frequent byte pairs. Café — 3.14 — 日本語 — 🦊."
    ) * 20
    t = BPETokenizer()
    t.train(sample, vocab_size=350, verbose=False)
    s = "The lazy dog éats a 🦊 in Café 日本."
    ids = t.encode(s)
    print(f"text      : {s}")
    print(f"tokens    : {ids}")
    print(f"n_tokens  : {len(ids)}  vs  {len(s)} chars, {len(s.encode('utf-8'))} bytes")
    print(f"roundtrip : {t.decode(ids) == s}")
