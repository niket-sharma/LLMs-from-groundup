# LLMs from the Ground Up

[![CI](https://github.com/niket-sharma/LLMs-from-groundup/actions/workflows/ci.yml/badge.svg)](https://github.com/niket-sharma/LLMs-from-groundup/actions/workflows/ci.yml)

A hands-on AI Engineer curriculum built around a GPT implemented from scratch
in PyTorch — from tokenizers and attention up through inference optimization,
modern architectures, scaling, post-training, and serving. Every concept ships
as a heavily-commented implementation, a correctness test against a naive
reference, and a benchmark that proves the claimed win.

The full roadmap lives in [`llm-engineer-enhancement.md`](llm-engineer-enhancement.md).

## Curriculum Map

| Module | Topic | Covers | Status |
|---|---|---|---|
| **Phase 0** | Foundation | Config-driven model factory, presets, CI, benchmarking harness | ✅ |
| **M1** | `modules/m1_fundamentals/` | BPE tokenizer from scratch, sinusoidal/RoPE, RMSNorm, SwiGLU, attention anatomy | ✅ |
| **M2** | `modules/m2_inference_opt/` | KV cache, MQA/GQA/MLA, FlashAttention, PagedAttention, continuous batching, speculative decoding | 🔜 |
| **M3** | `modules/m3_architectures/` | Mixture of Experts, Mamba/SSM, linear attention, diffusion LMs | 🔜 |
| **M4** | `modules/m4_training_scaling/` | Mixed precision, sequence packing, data pipelines, scaling laws | 🔜 |
| **M5** | `modules/m5_post_training/` | SFT, reward modeling, DPO, GRPO — from scratch on this repo's own GPT | 🔜 |
| **M6** | `modules/m6_serving/` | Quantization, distillation, FastAPI + continuous batching server, vLLM | 🔜 |

Library-based counterparts (TRL fine-tuning, LoRA, RLHF, API usage) live in
[`practical_llms/`](practical_llms/) — M5/M6 cross-link to them as the
"production" versions of the from-scratch implementations.

## Hardware Requirements

Everything runs on a single 8 GB NVIDIA GPU (WSL2/Linux, 32 GB RAM):
full-precision training up to ~124M params, quantized inference up to ~7B.
Tests and smoke benchmarks run CPU-only.

## Quick Start

```bash
pip install -e ".[dev]"   # core + test/lint tooling
make test                 # CPU test suite
make bench-smoke          # sanity benchmark (writes benchmarks/results/)
```

### Build a model

Models are config-driven; every architectural knob (positional encoding,
norm, activation, attention variant) is a `GPTConfig` field:

```python
import sys; sys.path.insert(0, "src")
from models import GPTConfig, create_model

model = create_model("tiny")                          # preset by name
model = create_model(GPTConfig(preset="base"))        # ~50M non-embedding params
model = create_model(GPTConfig(d_model=256, n_heads=8, vocab_size=5000))
```

Presets: `tiny` (~1M), `small` (~10M), `base` (~50M), `gpt2-ish` (~124M).

### Train and generate

```bash
python examples/train_example.py      # end-to-end char-level training demo
python examples/inference_example.py  # sampling strategies demo
```

```python
from models import create_model, GPTConfig
from training.dataset import prepare_data, create_dataloaders
from training.trainer import GPTTrainer

train_ds, val_ds, tokenizer = prepare_data("your_text.txt")
train_loader, val_loader = create_dataloaders(train_ds, val_ds)

model = create_model(GPTConfig(preset="tiny", vocab_size=tokenizer.vocab_size))
GPTTrainer(model, train_loader, val_loader).train(epochs=10, save_dir="checkpoints")
```

## Repo Structure

```
src/
├── models/        # GPT architecture: config.py, registry.py, attention, embeddings
├── training/      # tokenizer, dataset, trainer
└── utils/         # inference (sampling), helpers (seed, params, FLOPs)
modules/           # curriculum modules M1–M6 (added per the roadmap)
benchmarks/        # smoke.py + results/ (committed JSON/PNG)
practical_llms/    # library-based track: APIs, TRL fine-tuning, RLHF/DPO/GRPO
examples/          # runnable end-to-end scripts
tests/             # pytest suite (CPU, runs in CI)
visualizations/    # attention maps, embedding PCA, entropy analysis
```

Each `modules/mX_*/` folder follows a standard contract: `README.md`
(concepts + interview questions), a from-scratch implementation,
`benchmark.py`, and parity tests. **Golden rule:** prove correctness against
the naive reference first, then prove the win with committed benchmark
numbers.

## Development

```bash
make test          # pytest tests/ -q
make lint          # ruff check
make format        # ruff format + autofix
make bench-smoke   # tiny-model CPU benchmark
```

Agent/contributor conventions are in [`AGENTS.md`](AGENTS.md) (synced to
`CLAUDE.md`, `GEMINI.md`, etc. via `scripts/sync_agent_docs.sh`).

## References

- Attention Is All You Need (Vaswani et al., 2017)
- Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
- Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)

## License

MIT — see [LICENSE](LICENSE).
