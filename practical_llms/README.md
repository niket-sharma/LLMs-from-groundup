# Practical LLMs

This folder contains examples for working with pre-trained LLMs using APIs and the HuggingFace ecosystem. While the main `src/` folder teaches you how to build LLMs from scratch, this folder shows you how to use production-ready models.

## Structure

```
practical_llms/
├── inference/
│   ├── openai_api.py              # Using OpenAI API (GPT-4, GPT-3.5)
│   ├── anthropic_api.py           # Using Anthropic API (Claude)
│   └── huggingface_local.py       # Running HuggingFace models locally
│
├── finetuning/
│   ├── huggingface_sft.py         # Supervised Fine-Tuning with HuggingFace
│   ├── lora_finetuning.py         # Parameter-efficient fine-tuning with LoRA
│   ├── rlhf_trl.py                # RLHF using TRL library
│   ├── rl_for_llms_explained.py   # ⭐ Comprehensive RL tutorial (PPO, DPO)
│   └── rlvr_grpo_explained.py     # ⭐ RLVR & GRPO (DeepSeek-R1 technique)
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
| 7 | `finetuning/rl_for_llms_explained.py` | ⭐ Deep dive into RL for LLM alignment |
| 8 | `finetuning/rlvr_grpo_explained.py` | ⭐ RLVR & GRPO (DeepSeek-R1 technique) |

## Reinforcement Learning for LLMs

The `finetuning/rl_for_llms_explained.py` file provides a comprehensive tutorial on how RL is used to train LLMs:

### What You'll Learn

1. **Why RL for LLMs?** - The alignment problem and why supervised learning isn't enough
2. **Core Concepts** - How RL terminology maps to LLM training
3. **PPO Training** - Hands-on implementation of the classic RLHF approach
4. **DPO (Direct Preference Optimization)** - The simpler, modern alternative
5. **Comparison** - When to use PPO vs DPO vs ORPO

### The RLHF Pipeline

```
Pre-trained LLM → SFT (Supervised Fine-Tuning) → RLHF/DPO → Aligned Model
                  ↑                              ↑
                  Instruction data               Preference data
                  (instruction, response)        (chosen, rejected)
```

### Quick Example

```python
# PPO Training (classic RLHF)
from trl import PPOTrainer, PPOConfig

ppo_trainer = PPOTrainer(
    model=policy_model,
    ref_model=reference_model,
    config=PPOConfig(learning_rate=1e-5, kl_penalty="kl"),
)

# Training loop
for prompt in prompts:
    response = ppo_trainer.generate(prompt)
    reward = reward_model(response)
    ppo_trainer.step(prompt, response, reward)
```

```python
# DPO Training (simpler alternative)
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model=model,
    args=DPOConfig(beta=0.1),
    train_dataset=preference_dataset,  # (prompt, chosen, rejected)
)
trainer.train()
```

Run the full tutorial:
```bash
python practical_llms/finetuning/rl_for_llms_explained.py
```

## RLVR & GRPO: Training Reasoning Models (DeepSeek-R1)

The `finetuning/rlvr_grpo_explained.py` file covers the techniques used to train reasoning models like DeepSeek-R1:

### What is RLVR?

**RLVR (Reinforcement Learning with Verifiable Rewards)** uses programmatic verification instead of learned reward models. For tasks like math and code, we can VERIFY correctness directly:

| Task | Verification Method |
|------|---------------------|
| Math | Check if answer equals ground truth |
| Code | Run unit tests |
| Logic | Verify logical consistency |

### What is GRPO?

**GRPO (Group Relative Policy Optimization)** is a simpler alternative to PPO:

- No value network needed
- Advantage = reward - group_mean
- Sample multiple responses, reward correct ones
- The algorithm behind DeepSeek-R1's success

```python
# GRPO: Sample a group, compute relative advantages
responses = generate_group(prompt, n=4)
rewards = [verifier(r, ground_truth) for r in responses]
advantages = [r - mean(rewards) for r in rewards]  # Group-relative!
update_policy(responses, advantages)
```

### Key Insight

Models trained with RLVR spontaneously develop:
- ✓ Chain-of-thought reasoning
- ✓ Self-verification ("let me check...")
- ✓ Backtracking ("wait, that's wrong...")

All from just rewarding correct final answers!

Run the tutorial:
```bash
python practical_llms/finetuning/rlvr_grpo_explained.py
```

## Comparison: From Scratch vs Practical

| Aspect | From Scratch (`src/`) | Practical (`practical_llms/`) |
|--------|----------------------|------------------------------|
| **Purpose** | Learn internals | Get things done |
| **Models** | Small, custom | Large, pre-trained |
| **Training** | Full from scratch | Fine-tuning only |
| **Hardware** | CPU-friendly | GPU recommended |
| **Use Case** | Education | Production |
