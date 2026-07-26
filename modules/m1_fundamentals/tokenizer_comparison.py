#!/usr/bin/env python3
"""
Tokenizer comparison benchmark: char vs from-scratch BPE vs tiktoken vs SentencePiece.

The central tradeoff a tokenizer makes is **vocab size vs sequence length**:
- char-level: tiny vocab (~100), but every character is a token, so sequences
  are long and attention (O(n^2)) is expensive.
- byte-level BPE: a mid-sized vocab (here 1k-8k) buys a large drop in sequence
  length by merging frequent byte pairs into single tokens.
- tiktoken (GPT-2, 50257) / SentencePiece: production tables trained on huge
  corpora; on a small in-domain corpus they may or may not beat our tiny BPE,
  which is itself an instructive result.

We measure **compression** (bytes-per-token; higher = fewer tokens for the same
text) and vocab size, then plot the frontier. This is the module's benchmark:
it commits JSON + PNG to benchmarks/results/ per repo policy.

CLI guardrails: runs on CPU only (no model), tiny/fast by default. `--full`
sweeps more BPE vocab sizes; optional libs are used only if importable.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Import the from-scratch tokenizer whether run as a module or a file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m1_fundamentals.bpe_tokenizer import BPETokenizer  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "results"
DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "sample_text.txt"


def char_stats(text: str):
    """Char-level tokenizer stats: vocab = unique chars, one token per char."""
    vocab = sorted(set(text))
    n_tokens = len(text)
    return {
        "tokenizer": "char",
        "vocab_size": len(vocab),
        "n_tokens": n_tokens,
    }


def bpe_stats(text: str, vocab_size: int):
    t = BPETokenizer()
    t0 = time.perf_counter()
    t.train(text, vocab_size=vocab_size)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ids = t.encode(text)
    encode_s = time.perf_counter() - t0
    assert t.decode(ids) == text, "BPE must round-trip the corpus"
    return {
        "tokenizer": f"bpe-{vocab_size}",
        "vocab_size": t.vocab_size,
        "n_tokens": len(ids),
        "train_s": round(train_s, 3),
        "encode_s": round(encode_s, 4),
    }


def tiktoken_stats(text: str, name: str = "gpt2"):
    try:
        import tiktoken
    except ImportError:
        return None
    enc = tiktoken.get_encoding(name)
    ids = enc.encode(text)
    return {
        "tokenizer": f"tiktoken-{name}",
        "vocab_size": enc.n_vocab,
        "n_tokens": len(ids),
    }


def sentencepiece_stats(text: str, vocab_size: int):
    try:
        import sentencepiece as spm
    except ImportError:
        return None
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        corpus_path = Path(d) / "corpus.txt"
        corpus_path.write_text(text, encoding="utf-8")
        model_prefix = str(Path(d) / "sp")
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            # Small corpora need these relaxed or training refuses to run.
            character_coverage=0.9995,
            train_extremely_large_corpus=False,
        )
        sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
        ids = sp.encode(text)
    return {
        "tokenizer": f"sentencepiece-{vocab_size}",
        "vocab_size": vocab_size,
        "n_tokens": len(ids),
    }


def enrich(row: dict, n_chars: int, n_bytes: int) -> dict:
    """Add derived compression metrics to a stats row."""
    row["chars_per_token"] = round(n_chars / row["n_tokens"], 3)
    row["bytes_per_token"] = round(n_bytes / row["n_tokens"], 3)
    # Sequence-length reduction vs char-level (== n_chars tokens).
    row["seq_len_vs_char"] = round(row["n_tokens"] / n_chars, 3)
    return row


def plot(rows, out_png: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    labels = [r["tokenizer"] for r in rows]
    bytes_per_tok = [r["bytes_per_token"] for r in rows]
    vocab = [r["vocab_size"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(labels, bytes_per_tok, color="#4C72B0")
    ax1.set_ylabel("bytes / token  (higher = better compression)")
    ax1.set_title("Compression: fewer tokens per byte")
    ax1.tick_params(axis="x", rotation=30)

    ax2.scatter(vocab, bytes_per_tok, color="#C44E52")
    for lbl, x, y in zip(labels, vocab, bytes_per_tok):
        ax2.annotate(lbl, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax2.set_xscale("log")
    ax2.set_xlabel("vocab size (log)")
    ax2.set_ylabel("bytes / token")
    ax2.set_title("Vocab size vs compression frontier")

    fig.suptitle("Tokenizer comparison on the same corpus", fontweight="bold")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--device", default="cpu", help="unused (no model); kept for CLI parity")
    p.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=[512],
        help="BPE vocab sizes to sweep (default: one fast size)",
    )
    p.add_argument("--full", action="store_true", help="sweep 512/1024/2048/4096 and add sentencepiece")
    args = p.parse_args()

    text = Path(args.corpus).read_text(encoding="utf-8")
    n_chars = len(text)
    n_bytes = len(text.encode("utf-8"))

    vocab_sizes = [512, 1024, 2048, 4096] if args.full else args.vocab_sizes

    rows = [char_stats(text)]
    for vs in vocab_sizes:
        rows.append(bpe_stats(text, vs))
    tt = tiktoken_stats(text)
    if tt:
        rows.append(tt)
    if args.full:
        sp = sentencepiece_stats(text, min(vocab_sizes[-1], 2000))
        if sp:
            rows.append(sp)

    for r in rows:
        enrich(r, n_chars, n_bytes)

    result = {
        "benchmark": "tokenizer_comparison",
        "corpus": str(Path(args.corpus).name),
        "corpus_chars": n_chars,
        "corpus_bytes": n_bytes,
        "rows": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "m1_tokenizer_comparison.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    png_path = plot(rows, RESULTS_DIR / "m1_tokenizer_comparison.png")

    # Human-readable table.
    print(f"corpus: {result['corpus']}  ({n_chars} chars, {n_bytes} bytes)\n")
    hdr = f"{'tokenizer':<22}{'vocab':>8}{'tokens':>10}{'bytes/tok':>12}{'seq vs char':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['tokenizer']:<22}{r['vocab_size']:>8}{r['n_tokens']:>10}"
            f"{r['bytes_per_token']:>12}{r['seq_len_vs_char']:>14}"
        )
    print(f"\nwritten to {json_path}")
    if png_path:
        print(f"plot     {png_path}")


if __name__ == "__main__":
    main()
