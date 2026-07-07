#!/usr/bin/env python3
"""
Smoke benchmark: build a preset model, run a forward pass and a short
generation, record params/FLOPs/tokens-per-second to benchmarks/results/.

This is the CI sanity check (`make bench-smoke`) and the template every
module's benchmark.py follows: --device cpu and a tiny default so it always
runs on CI; heavier settings behind explicit flags.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models import GPTConfig, create_model  # noqa: E402
from utils.helpers import count_params, estimate_flops, set_seed  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="tiny", choices=["tiny", "small", "base", "gpt2-ish"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    config = GPTConfig(preset=args.preset, vocab_size=args.vocab_size)
    model = create_model(config).to(device).eval()

    total = count_params(model)
    non_emb = count_params(model, non_embedding=True)
    flops = estimate_flops(model, seq_len=args.seq_len)

    # Forward pass timing
    tokens = torch.randint(0, args.vocab_size, (1, args.seq_len), device=device)
    with torch.no_grad():
        model(tokens)  # warmup
        t0 = time.perf_counter()
        model(tokens)
        fwd_s = time.perf_counter() - t0

    # Generation timing (no KV cache yet — M2.1 will make this comparison
    # interesting; this number is the "before" baseline)
    prompt = tokens[:, : min(8, args.seq_len)]
    t0 = time.perf_counter()
    model.generate(prompt, max_new_tokens=args.gen_tokens)
    gen_s = time.perf_counter() - t0

    result = {
        "benchmark": "smoke",
        "preset": args.preset,
        "device": args.device,
        "vocab_size": args.vocab_size,
        "params_total": total,
        "params_non_embedding": non_emb,
        "flops_per_token_fwd": flops["flops_per_token_fwd"],
        "seq_len": args.seq_len,
        "forward_pass_s": round(fwd_s, 4),
        "gen_tokens": args.gen_tokens,
        "gen_tokens_per_s": round(args.gen_tokens / gen_s, 2),
        "torch_version": torch.__version__,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"smoke_{args.preset}_{args.device}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    print(f"\nwritten to {out_path.relative_to(Path.cwd()) if out_path.is_relative_to(Path.cwd()) else out_path}")


if __name__ == "__main__":
    main()
