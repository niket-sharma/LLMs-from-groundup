"""
Correctness tests for the from-scratch byte-level BPE tokenizer.

Golden rule (repo policy): prove correctness first. The headline guarantee of a
byte-level tokenizer is *lossless round-tripping of arbitrary UTF-8* — that's the
first and most important test. We also pin down merge determinism, save/load
fidelity, special-token handling, and a parity check against `tiktoken`'s
byte-level behavior where applicable.
"""

import os
import sys

import pytest

# Make the module importable both as a package and when run file-directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from m1_fundamentals.bpe_tokenizer import BPETokenizer, _merge, _get_stats  # noqa: E402


# A small but linguistically varied corpus for training in tests.
CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells sea shells by the sea shore. "
    "Peter Piper picked a peck of pickled peppers. "
    "1234567890 and 3.14159 and -42. "
    "Café naïve résumé — coöperate. 日本語のテキスト。 Русский текст. 🦊🚀✨"
) * 12


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=400)
    return t


# --------------------------------------------------------------- round-tripping
@pytest.mark.parametrize(
    "s",
    [
        "",
        "a",
        "hello world",
        "The lazy dog.",
        "   leading and trailing   ",
        "tabs\tand\nnewlines\r\n",
        "emoji 🦊🚀 and CJK 日本語 and Cyrillic Привет",
        "numbers 1234 and 3.14159 mixed with words",
        "Café naïve résumé coöperate",
        "\x00\x01\x02 control bytes \x7f",
        "�é",  # combining accent + surrogate-adjacent codepoints
    ],
)
def test_roundtrip_arbitrary_utf8(tok, s):
    """decode(encode(s)) == s for any UTF-8 string — the byte-level guarantee."""
    assert tok.decode(tok.encode(s)) == s


def test_roundtrip_on_untrained_bytes(tok):
    """Strings full of byte sequences never seen in training still round-trip."""
    weird = "".join(chr(c) for c in range(0x400, 0x500))  # a Cyrillic block
    assert tok.decode(tok.encode(weird)) == weird


def test_untrained_tokenizer_is_identity_over_bytes():
    """With no merges, encode == the raw UTF-8 byte values."""
    t = BPETokenizer()
    s = "hi 🦊"
    assert t.encode(s) == list(s.encode("utf-8"))
    assert t.decode(t.encode(s)) == s


# ------------------------------------------------------------------ compression
def test_merges_reduce_token_count(tok):
    """Trained BPE must produce fewer tokens than the raw byte stream."""
    text = "The quick brown fox jumps over the lazy dog. " * 5
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tok.encode(text))
    assert n_tokens < n_bytes, "BPE should compress vs raw bytes on in-domain text"


def test_vocab_size_matches_request():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=384)
    # 256 byte base + (384-256) merges == 384 entries in vocab.
    assert len(t.vocab) == 384
    assert t.vocab_size == 384  # no specials registered yet


# ------------------------------------------------------------------ determinism
def test_encoding_is_deterministic(tok):
    s = "deterministic encoding of the same string"
    assert tok.encode(s) == tok.encode(s)


def test_training_is_deterministic():
    a = BPETokenizer()
    a.train(CORPUS, vocab_size=400)
    b = BPETokenizer()
    b.train(CORPUS, vocab_size=400)
    assert a.merges == b.merges


# ----------------------------------------------------------------- merge helper
def test_merge_helper():
    # Replace pair (1, 2) with 99 in a small list.
    assert _merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    # No occurrences -> unchanged.
    assert _merge([1, 3, 4], (1, 2), 99) == [1, 3, 4]


def test_stats_helper():
    stats = _get_stats([1, 2, 1, 2, 3])
    assert stats[(1, 2)] == 2
    assert stats[(2, 1)] == 1
    assert stats[(2, 3)] == 1


# ------------------------------------------------------------------- save/load
def test_save_load_roundtrip(tok, tmp_path):
    path = str(tmp_path / "tok.json")
    tok.save(path)
    reloaded = BPETokenizer.load(path)
    assert reloaded.merges == tok.merges
    assert reloaded.vocab == tok.vocab
    s = "reloaded tokenizer must encode identically 🦊 日本"
    assert reloaded.encode(s) == tok.encode(s)
    assert reloaded.decode(reloaded.encode(s)) == s


# --------------------------------------------------------------- special tokens
def test_special_tokens():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=400)
    t.register_special_tokens(["<|endoftext|>", "<|user|>"])
    eot = t.special_tokens["<|endoftext|>"]

    ids = t.encode("hello<|endoftext|>world")
    assert eot in ids
    # The special token id is a single token, not BPE-split.
    assert ids.count(eot) == 1
    assert t.decode(ids) == "hello<|endoftext|>world"

    # allowed_special="none" treats the marker as literal text.
    ids_literal = t.encode("hello<|endoftext|>world", allowed_special="none")
    assert eot not in ids_literal
    assert t.decode(ids_literal) == "hello<|endoftext|>world"


# ---------------------------------------------------- parity vs tiktoken (optional)
def test_parity_vs_tiktoken_roundtrip():
    """
    We can't match GPT-2's exact merge table without its training data, but we
    can assert the *contract* both share: byte-level round-tripping. If tiktoken
    is installed, confirm our decode(encode(...)) agrees with identity on the
    same strings tiktoken round-trips.
    """
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=500)
    for s in ["Hello, world!", " leading space", "日本語 🦊", "num 42.0"]:
        assert enc.decode(enc.encode(s)) == s          # tiktoken round-trips
        assert t.decode(t.encode(s)) == s              # ours round-trips too


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
