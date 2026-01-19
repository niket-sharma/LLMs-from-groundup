#!/usr/bin/env python3
"""
RLHF with TRL (Transformer Reinforcement Learning) Library
==========================================================

This tutorial shows how to do RLHF using HuggingFace's TRL library,
which provides production-ready implementations of:

1. SFT (Supervised Fine-Tuning) Trainer
2. Reward Model Training
3. PPO (Proximal Policy Optimization) Trainer
4. DPO (Direct Preference Optimization) - simpler alternative to PPO

TRL handles all the complexity of RLHF for you!

Prerequisites:
    pip install trl transformers datasets peft accelerate
"""

import os
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


def check_dependencies():
    """Check if required packages are installed."""
    required = ['trl', 'transformers', 'datasets', 'peft']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {missing}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


# =============================================================================
# Part 1: SFT with TRL (Simplified)
# =============================================================================

def sft_with_trl():
    """
    Supervised Fine-Tuning using TRL's SFTTrainer.

    SFTTrainer simplifies fine-tuning by:
    - Auto-formatting conversation data
    - Handling chat templates
    - Built-in packing for efficiency
    """
    print("\n" + "=" * 60)
    print("Part 1: SFT with TRL's SFTTrainer")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    # Create sample dataset
    data = [
        {"text": "<|user|>\nWhat is Python?\n<|assistant|>\nPython is a high-level programming language known for readability and versatility."},
        {"text": "<|user|>\nExplain machine learning.\n<|assistant|>\nMachine learning is AI that learns patterns from data without explicit programming."},
        {"text": "<|user|>\nWhat is an API?\n<|assistant|>\nAn API is a set of protocols allowing software applications to communicate with each other."},
    ] * 30

    dataset = Dataset.from_list(data)

    # Load model
    model_name = "distilgpt2"
    print(f"\nLoading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # SFT Config
    sft_config = SFTConfig(
        output_dir="./sft_trl_output",
        max_seq_length=256,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        logging_steps=10,
        report_to="none",
    )

    # Create SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nSFTTrainer ready!")
    print("To train: trainer.train()")

    # Show what the formatted data looks like
    print("\nSample formatted data:")
    print(data[0]["text"][:200])

    return trainer


# =============================================================================
# Part 2: Reward Model Training with TRL
# =============================================================================

def train_reward_model():
    """
    Train a reward model using TRL's RewardTrainer.

    The reward model learns human preferences from comparison data:
    - Given two responses, which is better?
    - Outputs a scalar reward for any response
    """
    print("\n" + "=" * 60)
    print("Part 2: Reward Model Training")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from trl import RewardConfig, RewardTrainer
    from datasets import Dataset

    # Preference data format:
    # - "chosen": the preferred response
    # - "rejected": the less preferred response
    preference_data = [
        {
            "chosen": "Python is a high-level programming language known for its clear syntax and readability. It's widely used in web development, data science, and AI.",
            "rejected": "python is a language i guess, its ok",
        },
        {
            "chosen": "Machine learning is a subset of AI where systems learn from data to improve their performance without being explicitly programmed.",
            "rejected": "ml is when computers do stuff automatically",
        },
        {
            "chosen": "To reverse a string in Python, you can use slicing: text[::-1]. This creates a new string with characters in reverse order.",
            "rejected": "just use reverse i think",
        },
    ] * 30

    dataset = Dataset.from_list(preference_data)

    # Load model for sequence classification (outputs single score)
    model_name = "distilgpt2"
    print(f"\nLoading reward model base: {model_name}")

    # Note: For reward models, we typically use AutoModelForSequenceClassification
    # with num_labels=1 (single scalar output)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # Single reward score
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Reward training config
    reward_config = RewardConfig(
        output_dir="./reward_model_output",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=1e-5,
        logging_steps=10,
        max_length=256,
        report_to="none",
    )

    # Create RewardTrainer
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    print("\nRewardTrainer ready!")
    print("To train: trainer.train()")

    print("\nPreference data format:")
    print(f"  Chosen: '{preference_data[0]['chosen'][:50]}...'")
    print(f"  Rejected: '{preference_data[0]['rejected'][:50]}...'")

    return trainer


# =============================================================================
# Part 3: PPO Training with TRL
# =============================================================================

def ppo_training():
    """
    PPO training for RLHF using TRL.

    This is the "classic" RLHF approach:
    1. Generate responses with policy model
    2. Score with reward model
    3. Update policy using PPO
    """
    print("\n" + "=" * 60)
    print("Part 3: PPO Training (RLHF)")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    from datasets import Dataset

    model_name = "distilgpt2"
    print(f"\nLoading models: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Policy model (with value head for PPO)
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    # Reference model (frozen, for KL penalty)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    # Reward model (in practice, use a trained one)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
    )

    # PPO Config
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        ppo_epochs=4,
        kl_penalty="kl",
        target_kl=0.1,
        log_with=None,
    )

    # Create dataset of prompts
    prompts = [
        "Explain what Python is:",
        "What is machine learning?",
        "How do computers work?",
        "What is the internet?",
    ] * 10

    dataset = Dataset.from_dict({"query": prompts})

    # Tokenize prompts
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"])
        return sample

    dataset = dataset.map(tokenize)

    # Create PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        processing_class=tokenizer,
    )

    print("\nPPOTrainer ready!")
    print("\nPPO training loop (simplified):")

    ppo_loop_code = '''
    for batch in dataloader:
        # 1. Generate responses
        response_tensors = ppo_trainer.generate(batch["input_ids"])

        # 2. Get rewards from reward model
        rewards = reward_model(response_tensors)

        # 3. PPO update
        stats = ppo_trainer.step(
            batch["input_ids"],
            response_tensors,
            rewards
        )

        print(f"Mean reward: {stats['ppo/mean_scores']:.2f}")
    '''
    print(ppo_loop_code)

    return ppo_trainer


# =============================================================================
# Part 4: DPO (Direct Preference Optimization) - Simpler Alternative
# =============================================================================

def dpo_training():
    """
    DPO training - a simpler alternative to PPO.

    DPO skips the reward model entirely!
    - Directly optimizes policy from preference pairs
    - No reward model needed
    - No complex RL training loop
    - Often works just as well or better than PPO

    The key insight: You can derive an optimal policy directly from
    preference data without explicitly learning a reward function.
    """
    print("\n" + "=" * 60)
    print("Part 4: DPO (Direct Preference Optimization)")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import DPOConfig, DPOTrainer
    from datasets import Dataset

    model_name = "distilgpt2"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # DPO uses preference pairs directly
    # Format: prompt + chosen vs prompt + rejected
    dpo_data = [
        {
            "prompt": "What is Python?",
            "chosen": "Python is a versatile, high-level programming language known for its readable syntax. It's widely used in web development, data science, machine learning, and automation.",
            "rejected": "its a programming thing",
        },
        {
            "prompt": "Explain recursion.",
            "chosen": "Recursion is when a function calls itself to solve a problem by breaking it into smaller subproblems. It requires a base case to prevent infinite loops.",
            "rejected": "when function call itself",
        },
        {
            "prompt": "What is an API?",
            "chosen": "An API (Application Programming Interface) defines how software components interact. It specifies methods, data formats, and conventions for communication between systems.",
            "rejected": "api is for connecting stuff",
        },
    ] * 30

    dataset = Dataset.from_list(dpo_data)

    # DPO Config
    dpo_config = DPOConfig(
        output_dir="./dpo_output",
        beta=0.1,  # KL penalty coefficient
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=5e-6,
        logging_steps=10,
        max_length=256,
        max_prompt_length=128,
        report_to="none",
    )

    # Create DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nDPOTrainer ready!")
    print("To train: trainer.train()")

    print("\nDPO vs PPO comparison:")
    comparison = """
    ┌─────────────────┬────────────────────┬──────────────────────┐
    │ Aspect          │ PPO (RLHF)         │ DPO                  │
    ├─────────────────┼────────────────────┼──────────────────────┤
    │ Reward Model    │ Required           │ Not needed           │
    │ Training        │ Complex (RL loop)  │ Simple (supervised)  │
    │ Stability       │ Can be unstable    │ Very stable          │
    │ Memory          │ 4 models needed    │ 2 models needed      │
    │ Performance     │ Good               │ Often better         │
    │ Implementation  │ Complex            │ Simple               │
    └─────────────────┴────────────────────┴──────────────────────┘
    """
    print(comparison)

    return trainer


# =============================================================================
# Part 5: ORPO (Odds Ratio Preference Optimization) - Even Simpler
# =============================================================================

def orpo_training():
    """
    ORPO - An even simpler preference optimization method.

    ORPO combines SFT and preference optimization in one step:
    - No reference model needed
    - Single training phase
    - Very memory efficient
    """
    print("\n" + "=" * 60)
    print("Part 5: ORPO (Odds Ratio Preference Optimization)")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import ORPOConfig, ORPOTrainer
    from datasets import Dataset

    model_name = "distilgpt2"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Same format as DPO
    orpo_data = [
        {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a field of AI where algorithms learn patterns from data to make predictions or decisions without explicit programming.",
            "rejected": "ml is computer learning stuff",
        },
        {
            "prompt": "Explain cloud computing.",
            "chosen": "Cloud computing delivers computing resources (servers, storage, databases, networking) over the internet on a pay-as-you-go basis.",
            "rejected": "computers in the cloud",
        },
    ] * 30

    dataset = Dataset.from_list(orpo_data)

    # ORPO Config
    orpo_config = ORPOConfig(
        output_dir="./orpo_output",
        beta=0.1,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=5e-6,
        logging_steps=10,
        max_length=256,
        max_prompt_length=128,
        report_to="none",
    )

    # Create ORPO Trainer
    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nORPOTrainer ready!")
    print("To train: trainer.train()")

    print("\nORPO advantages:")
    print("  - No reference model (saves memory)")
    print("  - Single-stage training")
    print("  - Combines SFT + alignment")
    print("  - State-of-the-art results")

    return trainer


# =============================================================================
# Part 6: Complete RLHF Pipeline with TRL
# =============================================================================

def complete_pipeline():
    """
    Show the complete RLHF pipeline structure.
    """
    print("\n" + "=" * 60)
    print("Complete RLHF Pipeline with TRL")
    print("=" * 60)

    pipeline = """
    ┌─────────────────────────────────────────────────────────────┐
    │                    RLHF Pipeline                             │
    └─────────────────────────────────────────────────────────────┘

    Step 1: Pre-train or use existing base model
    ─────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained("llama-7b")

    Step 2: Supervised Fine-Tuning (SFT)
    ─────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=instruction_dataset,  # (instruction, response) pairs
    )
    trainer.train()
    # Model now follows instructions

    Step 3a: Reward Model Training (for PPO)
    ────────────────────────────────────────
    trainer = RewardTrainer(
        model=reward_model,
        train_dataset=preference_dataset,  # (chosen, rejected) pairs
    )
    trainer.train()
    # Reward model scores response quality

    Step 3b: PPO Training (classic RLHF)
    ────────────────────────────────────
    ppo_trainer = PPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
    )
    # Train with RL to maximize reward

    OR

    Step 3c: DPO Training (simpler alternative)
    ──────────────────────────────────────────
    trainer = DPOTrainer(
        model=sft_model,
        train_dataset=preference_dataset,
    )
    trainer.train()
    # Directly optimize from preferences, no reward model!

    Step 4: Evaluation and Deployment
    ─────────────────────────────────
    model.save_pretrained("./final_model")
    model.push_to_hub("my-aligned-model")

    ┌─────────────────────────────────────────────────────────────┐
    │                    Recommended Approach                      │
    └─────────────────────────────────────────────────────────────┘

    For most use cases, use DPO or ORPO instead of PPO:
    - Simpler to implement
    - More stable training
    - No reward model needed
    - Often better results

    PPO is mainly useful when:
    - You need online learning
    - You want to use a separately trained reward model
    - You're doing research on RL methods
    """
    print(pipeline)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the TRL RLHF tutorial."""
    print("=" * 60)
    print("RLHF with TRL (Transformer Reinforcement Learning)")
    print("=" * 60)

    if not check_dependencies():
        return

    try:
        sft_with_trl()
        train_reward_model()
        ppo_training()
        dpo_training()
        orpo_training()
        complete_pipeline()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print("""
    Summary:
    --------
    TRL provides production-ready implementations of:

    1. SFTTrainer - Supervised fine-tuning
    2. RewardTrainer - Train reward models
    3. PPOTrainer - Classic RLHF with PPO
    4. DPOTrainer - Direct preference optimization (recommended!)
    5. ORPOTrainer - Single-stage preference optimization

    Recommended path:
    1. Start with SFT to teach instruction following
    2. Use DPO or ORPO for alignment (simpler than PPO)
    3. Use LoRA/QLoRA for memory efficiency

    Resources:
    - TRL docs: https://huggingface.co/docs/trl
    - DPO paper: https://arxiv.org/abs/2305.18290
    - ORPO paper: https://arxiv.org/abs/2403.07691
    """)


if __name__ == "__main__":
    main()
