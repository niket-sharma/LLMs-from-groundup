#!/usr/bin/env python3
"""KV-cache throughput benchmark; CPU-safe by default, GPU optional."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Match the repository's runnable example/module convention.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from models import GPTConfig, create_model


def generate_naive(model, prompt, new_tokens):
    tokens = prompt
    with torch.no_grad():
        for _ in range(new_tokens):
            logits = model(tokens[:, -model.max_seq_len:])
            tokens = torch.cat((tokens, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
    return tokens


def generate_cached(model, prompt, new_tokens):
    tokens, current, past = prompt, prompt, None
    with torch.no_grad():
        for _ in range(new_tokens):
            logits, past = model(current, past_key_values=past, use_cache=True)
            current = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, current), dim=1)
    return tokens


def timed(fn, warmup=1, runs=3):
    for _ in range(warmup):
        fn()
    started = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - started) / runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.full:
        args.preset, args.prompt_len, args.new_tokens = "base", 256, 128
    torch.manual_seed(0)
    model = create_model(GPTConfig(preset=args.preset, vocab_size=256, max_seq_len=max(256, args.prompt_len + args.new_tokens), dropout=0.0, pos_encoding="rope")).to(args.device).eval()
    prompt = torch.randint(0, 256, (1, args.prompt_len), device=args.device)
    naive_out, cached_out = generate_naive(model, prompt, args.new_tokens), generate_cached(model, prompt, args.new_tokens)
    assert torch.equal(naive_out, cached_out), "cached greedy output must match naive output"
    naive_s = timed(lambda: generate_naive(model, prompt, args.new_tokens))
    cached_s = timed(lambda: generate_cached(model, prompt, args.new_tokens))
    result = {"benchmark": "m2_kv_cache", "preset": args.preset, "device": args.device,
              "prompt_len": args.prompt_len, "new_tokens": args.new_tokens,
              "naive_tokens_per_s": round(args.new_tokens / naive_s, 3),
              "cached_tokens_per_s": round(args.new_tokens / cached_s, 3),
              "speedup": round(naive_s / cached_s, 3), "token_identical": True}
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "results" / "m2_kv_cache.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
