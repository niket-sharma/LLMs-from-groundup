# Practical LLMs

This folder contains examples for working with pre-trained LLMs using APIs and the HuggingFace ecosystem. While the main `src/` folder teaches you how to build LLMs from scratch, this folder shows you how to use production-ready models.

## Structure

```
practical_llms/
├── inference/
│   ├── openai_api.py          # Using OpenAI API (GPT-4, GPT-3.5)
│   ├── anthropic_api.py       # Using Anthropic API (Claude)
│   ├── huggingface_local.py   # Running HuggingFace models locally
│   └── huggingface_api.py     # Using HuggingFace Inference API
│
├── finetuning/
│   ├── huggingface_sft.py     # Supervised Fine-Tuning with HuggingFace
│   ├── lora_finetuning.py     # Parameter-efficient fine-tuning with LoRA
│   └── rlhf_trl.py            # RLHF using TRL library
│
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install openai anthropic transformers datasets accelerate peft trl bitsandbytes
```

### 2. Set Up API Keys

```bash
# For OpenAI
export OPENAI_API_KEY="your-key-here"

# For Anthropic
export ANTHROPIC_API_KEY="your-key-here"

# For HuggingFace (optional, for gated models)
export HF_TOKEN="your-token-here"
```

### 3. Run Examples

```bash
# API-based inference
python practical_llms/inference/openai_api.py
python practical_llms/inference/anthropic_api.py

# Local inference with HuggingFace
python practical_llms/inference/huggingface_local.py

# Fine-tuning
python practical_llms/finetuning/huggingface_sft.py
```

## Learning Path

| Stage | File | What You'll Learn |
|-------|------|-------------------|
| 1 | `inference/openai_api.py` | Basic API calls, chat completions, streaming |
| 2 | `inference/anthropic_api.py` | Claude API, system prompts, tool use |
| 3 | `inference/huggingface_local.py` | Loading models, tokenization, generation |
| 4 | `finetuning/huggingface_sft.py` | Supervised fine-tuning on custom data |
| 5 | `finetuning/lora_finetuning.py` | Efficient fine-tuning with LoRA/QLoRA |
| 6 | `finetuning/rlhf_trl.py` | RLHF with the TRL library |

## Comparison: From Scratch vs Practical

| Aspect | From Scratch (`src/`) | Practical (`practical_llms/`) |
|--------|----------------------|------------------------------|
| **Purpose** | Learn internals | Get things done |
| **Models** | Small, custom | Large, pre-trained |
| **Training** | Full from scratch | Fine-tuning only |
| **Hardware** | CPU-friendly | GPU recommended |
| **Use Case** | Education | Production |
