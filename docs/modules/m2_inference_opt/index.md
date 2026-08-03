# M2 — Inference Optimization: KV Cache

## Understand it

Autoregressive decoding normally recomputes every transformer layer for the
whole prefix after each generated token. A KV cache stores each layer's projected,
position-encoded keys and values. The next query attends to those saved tensors,
so prior projections and blocks are not recomputed.

The implementation is in
[`modules/m2_inference_opt/kv_cache.py`](../../../modules/m2_inference_opt/kv_cache.py)
and is threaded through
[`src/models/attention.py`](../../../src/models/attention.py) and
[`src/models/gpt.py`](../../../src/models/gpt.py). The key invariant is that
cached incremental logits match a full causal forward pass within `1e-5`.

## Run it

<!-- run: cpu-smoke; cwd: repo-root; timeout: 60s -->
```bash
make m2-smoke
```

The CPU benchmark writes `benchmarks/results/m2_kv_cache.json`. GPU/full runs
are explicit: `python modules/m2_inference_opt/benchmark.py --device cuda --full`.

## Inspect and debug it

Start at `SmallGPT.forward(..., use_cache=True)`: it returns logits plus one
`(K, V)` tuple per transformer block. On the first call the prompt fills the
cache; subsequent calls pass a single token and append one position. If parity
fails, inspect cache sequence dimension `2`, the causal mask shape
`(B, 1, T_new, T_total)`, and positional offset handling for learned embeddings
and RoPE.

## Validate and test

`test_cached_logits_match_full_forward` is the parity proof;
`test_cached_greedy_generation_is_token_identical` validates public inference;
and the static/dynamic cache test verifies allocation strategies agree. The
benchmark asserts token identity before reporting tokens/sec. Speedup is
hardware- and sequence-length-dependent; the 5×/512-token acceptance target
requires the later base-model GPU run, not the CPU smoke result.

## Troubleshooting and learning checks

If generation diverges, use `dropout=0`, greedy decoding and the fixed test seed.
If a sequence exceeds `max_seq_len`, crop and start a fresh cache—cached absolute
positions cannot simply discard only the earliest K/V rows. Explain why V is
cached even though RoPE is applied only to Q/K; then compare prefill cost with a
one-token decode step and identify which work caching removes.
