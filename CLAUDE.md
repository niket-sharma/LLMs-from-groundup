<!-- AUTO-GENERATED from AGENTS.md by scripts/sync_agent_docs.sh — edit AGENTS.md instead. -->

# Agent Instructions — LLMs-from-groundup

Canonical instruction file for all coding agents (Claude Code, Codex, Cursor,
Aider, Gemini CLI, Copilot). Edit **this file only**, then run
`scripts/sync_agent_docs.sh` to regenerate the per-harness copies.

## What this repo is

A hands-on AI Engineer curriculum built around a small GPT implemented from
scratch in PyTorch. The roadmap lives in `llm-engineer-enhancement.md`
(Phase 0 + modules M1–M6). Work proceeds one phase per branch per PR.

## Command surface

Always use the Make targets — they pick the right Python automatically:

- `make test` — full CPU test suite (must stay green)
- `make lint` — ruff check on `src`, `tests`, `benchmarks`
- `make format` — ruff format + autofix
- `make bench-smoke` — fast CPU sanity benchmark (< 1 min)
- `make docs-check` — validates module run/understand/validate/test guides

## Architecture rules

- **Config-driven models:** all models are built via
  `create_model(GPTConfig(...))` from `src/models/registry.py`. New
  architectural variants (RoPE, RMSNorm, GQA, ...) are added as `GPTConfig`
  fields + registry support, never as separate ad-hoc model classes.
- **Presets:** `tiny` (~1M), `small` (~10M), `base` (~50M), `gpt2-ish`
  (~124M). Experiments use presets so results are comparable.
- **Backward compatibility:** `create_small_gpt(dict)` and the existing
  `examples/` scripts must keep working.

## Module folder contract (`modules/mX_topic/`)

Each module ships code, evidence, and a learning guide:

1. `README.md` — concepts, math, diagrams, interview questions
2. `<topic>.py` — heavily-commented from-scratch implementation
3. `benchmark.py` — measures the claimed improvement
4. `test_<topic>.py` — correctness tests (parity vs reference implementation)
5. `docs/modules/mX_topic/index.md` — how to understand, run, validate, test,
   and troubleshoot the implementation

**Golden rule:** parity test before benchmark. Prove correctness against the
naive reference (within tolerance), *then* prove the win. Benchmark outputs
(JSON + PNG) are committed to `benchmarks/results/`.
Documentation is updated in the same PR as the implementation; every documented
CPU-smoke command must be runnable from repository root.

## Hardware limits

8 GB VRAM (NVIDIA, WSL2), 32 GB RAM. Never default to models > 3B for GPU
work; full-precision training caps at ~124M params. Everything must also run
on CPU for CI.

## CLI guardrails

Every training/benchmark script accepts `--device cpu` and defaults to a
tiny, fast configuration; long/GPU runs sit behind explicit flags
(`--full`, `--preset base`). CI only ever runs the tiny CPU path.

## Library policy

- Pure PyTorch for from-scratch modules (`src/`, `modules/m1`–`m5`)
- TRL / bitsandbytes / vLLM / transformers only in `practical_llms/` and
  `modules/m6_serving/`

## Commit discipline

- Conventional commits, one logical change per commit
- No binary artifacts in git (checkpoints, tokenizer pickles, safetensors);
  benchmark JSON/PNG results in `benchmarks/results/` are the exception
- After implementing a phase, verify each acceptance checkbox in
  `llm-engineer-enhancement.md` against real command output before opening a PR
