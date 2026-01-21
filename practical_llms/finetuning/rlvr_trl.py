#!/usr/bin/env python3
"""
RLVR with TRL: Reinforcement Learning with Verifiable Rewards
==============================================================

This tutorial shows how to do RLVR using HuggingFace's TRL library,
demonstrating the techniques used to train reasoning models like DeepSeek-R1.

Key features:
1. GRPO (Group Relative Policy Optimization) - Simpler than PPO
2. Verifiable Rewards - Programmatic verification instead of learned reward models
3. Math/Code verification - Perfect reward signals for reasoning tasks

TRL provides GRPOTrainer which implements this algorithm!

Prerequisites:
    pip install trl transformers datasets accelerate

Usage:
    python rlvr_trl.py           # Show tutorial (no training)
    python rlvr_trl.py --train   # Actually run training (1 epoch each)
"""

import os
import sys
import argparse
import torch
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Global flag for whether to actually run training
RUN_TRAINING = False


def check_dependencies():
    """Check if required packages are installed."""
    required = ['trl', 'transformers', 'datasets']
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
# Part 1: Understanding RLVR vs RLHF
# =============================================================================

def explain_rlvr():
    """Explain the difference between RLVR and RLHF."""
    print("\n" + "=" * 60)
    print("Part 1: Understanding RLVR (Verifiable Rewards)")
    print("=" * 60)

    explanation = """
    RLVR vs RLHF:
    ─────────────

    RLHF (Traditional):
    ┌──────────┐      ┌──────────────┐      ┌──────────┐
    │  Policy  │ ──►  │ Reward Model │ ──►  │  Reward  │
    │  Output  │      │  (Learned)   │      │  Score   │
    └──────────┘      └──────────────┘      └──────────┘
                            ↑
                      Can be fooled!

    RLVR (Verifiable):
    ┌──────────┐      ┌──────────────┐      ┌──────────┐
    │  Policy  │ ──►  │   Verifier   │ ──►  │  Reward  │
    │  Output  │      │(Programmatic)│      │  (0 or 1)│
    └──────────┘      └──────────────┘      └──────────┘
                            ↑
                      Perfect signal!

    Verifiable Tasks:
    ┌─────────────────┬────────────────────────────────────┐
    │ Task            │ Verification Method                │
    ├─────────────────┼────────────────────────────────────┤
    │ Math Problems   │ Compare answer to ground truth     │
    │ Code Generation │ Run unit tests                     │
    │ Logic Puzzles   │ Check logical consistency          │
    │ SQL Queries     │ Execute and compare results        │
    └─────────────────┴────────────────────────────────────┘

    Key Insight from DeepSeek-R1:
    Models trained with RLVR spontaneously develop:
    ✓ Chain-of-thought reasoning
    ✓ Self-verification ("let me check...")
    ✓ Backtracking ("wait, that's wrong...")

    All from just rewarding correct final answers!
    """
    print(explanation)


# =============================================================================
# Part 2: Implementing Verifiers
# =============================================================================

class MathVerifier:
    """
    Verifier for math problems.
    Returns reward based on whether the answer is correct.
    """

    def __init__(self):
        self.answer_patterns = [
            r"(?:answer|result|equals|=)\s*[:\s]*(-?\d+\.?\d*)",
            r"(-?\d+\.?\d*)\s*$",
            r"####\s*(-?\d+\.?\d*)",
        ]

    def extract_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from text."""
        text = text.lower().strip()
        for pattern in self.answer_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None

    def __call__(self, response: str, ground_truth: float) -> float:
        """Return 1.0 if correct, 0.0 if incorrect."""
        extracted = self.extract_answer(response)
        if extracted is None:
            return 0.0
        return 1.0 if abs(extracted - ground_truth) < 1e-6 else 0.0


class CodeVerifier:
    """
    Verifier for code generation.
    Runs the code and checks if it produces expected output.
    """

    def __call__(self, code: str, test_cases: List[Tuple[str, str]]) -> float:
        """
        Run code against test cases.
        Returns fraction of tests passed.
        """
        passed = 0
        for input_val, expected_output in test_cases:
            try:
                # Create a safe execution environment
                local_vars = {}
                exec(code, {"__builtins__": {}}, local_vars)

                # Find the function and call it
                for name, obj in local_vars.items():
                    if callable(obj):
                        result = str(obj(eval(input_val)))
                        if result.strip() == expected_output.strip():
                            passed += 1
                        break
            except Exception:
                continue

        return passed / len(test_cases) if test_cases else 0.0


def demonstrate_verifiers():
    """Show how verifiers work."""
    print("\n" + "=" * 60)
    print("Part 2: Verifiers for RLVR")
    print("=" * 60)

    # Math verifier demo
    math_verifier = MathVerifier()

    print("\nMath Verifier Examples:")
    print("-" * 40)

    test_cases = [
        ("The answer is 42", 42.0),
        ("Let me calculate: 15 + 27 = 42", 42.0),
        ("#### 42", 42.0),
        ("I think it's 41", 42.0),  # Wrong
    ]

    for response, truth in test_cases:
        reward = math_verifier(response, truth)
        status = "✓ Correct" if reward > 0 else "✗ Wrong"
        print(f"  Response: '{response}'")
        print(f"  Ground truth: {truth}, Reward: {reward} {status}")
        print()

    print("\nCode Verifier Example:")
    print("-" * 40)

    code_verifier = CodeVerifier()

    # Example: function to add two numbers
    correct_code = """
def add(x, y):
    return x + y
"""
    wrong_code = """
def add(x, y):
    return x - y
"""

    test_cases_code = [("(2, 3)", "5"), ("(10, 20)", "30")]

    print(f"  Correct code reward: {code_verifier(correct_code, test_cases_code)}")
    print(f"  Wrong code reward: {code_verifier(wrong_code, test_cases_code)}")


# =============================================================================
# Part 3: GRPO with TRL
# =============================================================================

def grpo_training():
    """
    GRPO training using TRL's GRPOTrainer.

    GRPO (Group Relative Policy Optimization) is simpler than PPO:
    - No value network needed
    - Advantage = reward - group_mean
    - Works great with binary/sparse rewards
    """
    print("\n" + "=" * 60)
    print("Part 3: GRPO Training with TRL")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    model_name = "distilgpt2"
    print(f"\nLoading model: {model_name}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create math problem dataset
    # Format: prompt with the math problem
    math_problems = [
        {"prompt": "What is 5 + 3? The answer is", "answer": 8.0},
        {"prompt": "Calculate 10 - 4. The answer is", "answer": 6.0},
        {"prompt": "What is 7 * 2? The answer is", "answer": 14.0},
        {"prompt": "Compute 20 / 4. The answer is", "answer": 5.0},
        {"prompt": "What is 15 + 27? The answer is", "answer": 42.0},
        {"prompt": "Calculate 100 - 37. The answer is", "answer": 63.0},
        {"prompt": "What is 8 * 9? The answer is", "answer": 72.0},
        {"prompt": "Compute 144 / 12. The answer is", "answer": 12.0},
    ] * 5  # Repeat for more data

    # Create dataset with just prompts (answers used for verification)
    dataset = Dataset.from_list([{"prompt": p["prompt"]} for p in math_problems])

    # Store answers for reward computation
    answers = {p["prompt"]: p["answer"] for p in math_problems}

    # Create verifier
    math_verifier = MathVerifier()

    # Define reward function for GRPO
    def reward_function(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        """
        Compute rewards using math verifier.
        This is the key to RLVR - programmatic verification!
        """
        rewards = []
        for completion, prompt in zip(completions, prompts):
            ground_truth = answers.get(prompt, 0.0)
            reward = math_verifier(completion, ground_truth)
            rewards.append(reward)
        return rewards

    # GRPO Config
    grpo_config = GRPOConfig(
        output_dir="./grpo_output",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        num_generations=4,  # Sample 4 responses per prompt (group size)
        max_completion_length=32,  # Short completions for math
        max_prompt_length=64,
        logging_steps=5,
        report_to=None,
    )

    # Create GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function,
    )

    print("\nGRPOTrainer ready!")

    print("\nGRPO vs PPO comparison:")
    comparison = """
    ┌─────────────────┬──────────────────┬──────────────────┐
    │ Aspect          │ PPO              │ GRPO             │
    ├─────────────────┼──────────────────┼──────────────────┤
    │ Value Network   │ Required         │ Not needed       │
    │ Advantage       │ GAE (complex)    │ Group mean       │
    │ Memory          │ Higher           │ Lower            │
    │ Implementation  │ Complex          │ Simpler          │
    │ Best for        │ Dense rewards    │ Sparse/binary    │
    └─────────────────┴──────────────────┴──────────────────┘
    """
    print(comparison)

    if RUN_TRAINING:
        print("\n>>> Running GRPO training (1 epoch)...")
        trainer.train()
        print(">>> GRPO training complete!")
    else:
        print("To train: trainer.train()")

    return trainer


# =============================================================================
# Part 4: Online DPO with Verifiable Rewards
# =============================================================================

def online_dpo_with_verification():
    """
    Online DPO can also use verifiable rewards.
    Instead of pre-collected preferences, generate and verify on-the-fly.
    """
    print("\n" + "=" * 60)
    print("Part 4: Online DPO with Verifiable Rewards")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import OnlineDPOConfig, OnlineDPOTrainer
    from datasets import Dataset

    model_name = "distilgpt2"
    print(f"\nLoading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Math problems for online generation
    prompts = [
        "What is 5 + 3?",
        "Calculate 10 - 4.",
        "What is 7 * 2?",
        "Compute 20 / 4.",
    ] * 10

    dataset = Dataset.from_dict({"prompt": prompts})

    # Verifier-based reward function
    math_verifier = MathVerifier()
    ground_truths = {"What is 5 + 3?": 8, "Calculate 10 - 4.": 6,
                     "What is 7 * 2?": 14, "Compute 20 / 4.": 5}

    def reward_function(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        """Verify completions and return rewards."""
        rewards = []
        for completion, prompt in zip(completions, prompts):
            truth = ground_truths.get(prompt, 0)
            reward = math_verifier(completion, truth)
            rewards.append(reward)
        return rewards

    # Online DPO Config (TRL 0.27.0+ uses max_new_tokens instead of max_completion_length)
    online_dpo_config = OnlineDPOConfig(
        output_dir="./online_dpo_output",
        learning_rate=5e-6,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        max_new_tokens=32,  # Max tokens to generate
        max_length=96,  # Max total length (prompt + completion)
        logging_steps=5,
        report_to=None,
    )

    # Create Online DPO Trainer
    trainer = OnlineDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=online_dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function,
    )

    print("\nOnlineDPOTrainer ready!")

    print("\nOnline DPO with verification:")
    print("  1. Generate pairs of responses for each prompt")
    print("  2. Use verifier to determine which is 'chosen' vs 'rejected'")
    print("  3. Train with DPO loss on verified preferences")
    print("  4. No human labeling needed!")

    if RUN_TRAINING:
        print("\n>>> Running Online DPO training (1 epoch)...")
        trainer.train()
        print(">>> Online DPO training complete!")
    else:
        print("\nTo train: trainer.train()")

    return trainer


# =============================================================================
# Part 5: Complete RLVR Pipeline
# =============================================================================

def complete_rlvr_pipeline():
    """Show the complete RLVR pipeline."""
    print("\n" + "=" * 60)
    print("Part 5: Complete RLVR Pipeline")
    print("=" * 60)

    pipeline = """
    ┌─────────────────────────────────────────────────────────────┐
    │                    RLVR Pipeline                             │
    └─────────────────────────────────────────────────────────────┘

    Step 1: Prepare Verifiable Dataset
    ───────────────────────────────────
    # Math problems with ground truth answers
    dataset = [
        {"prompt": "What is 2+2?", "answer": 4},
        {"prompt": "Solve: 3x = 15", "answer": 5},
    ]

    Step 2: Create Verifier
    ───────────────────────
    def verify(response, ground_truth):
        extracted = extract_answer(response)
        return 1.0 if extracted == ground_truth else 0.0

    Step 3: Define Reward Function
    ──────────────────────────────
    def reward_fn(completions, prompts):
        return [verify(c, answers[p]) for c, p in zip(completions, prompts)]

    Step 4: Train with GRPO
    ───────────────────────
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,  # Verifiable rewards!
        train_dataset=dataset,
    )
    trainer.train()

    ┌─────────────────────────────────────────────────────────────┐
    │                    Why RLVR Works So Well                    │
    └─────────────────────────────────────────────────────────────┘

    1. Perfect Rewards
       - No reward model errors or biases
       - Cannot be hacked or gamed
       - Scales without human labeling

    2. Emergent Behaviors
       - Chain-of-thought reasoning appears naturally
       - Self-verification develops ("let me check...")
       - Error correction emerges ("wait, that's wrong...")

    3. Efficient Training
       - GRPO is simpler than PPO (no value network)
       - Binary rewards work great (correct/incorrect)
       - Fewer hyperparameters to tune

    ┌─────────────────────────────────────────────────────────────┐
    │                    Best Practices                            │
    └─────────────────────────────────────────────────────────────┘

    1. Start with SFT on reasoning examples (chain-of-thought)
    2. Use larger group sizes (4-8) for more stable training
    3. Mix difficulty levels in your dataset
    4. Consider process rewards (reward intermediate steps)
    5. Use temperature > 0 for diverse sampling
    """
    print(pipeline)


# =============================================================================
# Part 6: Comparison with other methods
# =============================================================================

def method_comparison():
    """Compare RLVR/GRPO with other methods."""
    print("\n" + "=" * 60)
    print("Part 6: Method Comparison")
    print("=" * 60)

    comparison = """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    RLVR Methods Comparison                               │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌───────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
    │ Method        │ Reward      │ Value Net   │ Complexity  │ Best For    │
    ├───────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
    │ PPO + RLHF    │ Learned RM  │ Required    │ High        │ General     │
    │ DPO           │ Implicit    │ No          │ Low         │ Preferences │
    │ GRPO + RLVR   │ Verifier    │ No          │ Medium      │ Math/Code   │
    │ Online DPO    │ Verifier    │ No          │ Medium      │ Math/Code   │
    │ ORPO          │ Implicit    │ No          │ Low         │ Single-stage│
    └───────────────┴─────────────┴─────────────┴─────────────┴─────────────┘

    When to use RLVR/GRPO:
    ─────────────────────
    ✓ Tasks with objectively correct answers (math, code, logic)
    ✓ When you can build a reliable verifier
    ✓ When you want to avoid reward model training
    ✓ When you need perfect reward signals

    When to use traditional RLHF:
    ────────────────────────────
    ✓ Subjective tasks (helpfulness, creativity, style)
    ✓ When correctness can't be programmatically verified
    ✓ When human preferences are the gold standard

    DeepSeek-R1 Recipe:
    ───────────────────
    1. Pre-train large base model
    2. Cold-start with small amount of CoT examples
    3. GRPO with math verifiers (RLVR)
    4. Rejection sampling for more training data
    5. Final SFT on curated outputs
    """
    print(comparison)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the TRL RLVR tutorial."""
    global RUN_TRAINING

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="RLVR Tutorial with TRL Library (Verifiable Rewards + GRPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rlvr_trl.py           # Show tutorial (no training)
  python rlvr_trl.py --train   # Actually run training (1 epoch each)
        """
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Actually run training (1 epoch for each method). Without this flag, only shows setup."
    )
    args = parser.parse_args()

    RUN_TRAINING = args.train

    print("=" * 60)
    print("RLVR with TRL (Verifiable Rewards + GRPO)")
    print("=" * 60)

    if RUN_TRAINING:
        print("\n*** TRAINING MODE: Will run actual training (1 epoch each) ***\n")
    else:
        print("\n*** DEMO MODE: Showing setup only. Use --train to run training ***\n")

    if not check_dependencies():
        return

    try:
        # Educational sections
        explain_rlvr()
        demonstrate_verifiers()

        # Training sections
        grpo_training()
        online_dpo_with_verification()

        # Summary sections
        complete_rlvr_pipeline()
        method_comparison()

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
    RLVR (RL with Verifiable Rewards) enables:

    1. Perfect reward signals for math/code tasks
    2. No reward model needed - use programmatic verification
    3. GRPO is simpler than PPO (no value network)
    4. Emergent reasoning behaviors (CoT, self-verification)

    Key TRL classes for RLVR:
    - GRPOTrainer: Group Relative Policy Optimization
    - OnlineDPOTrainer: Online DPO with custom rewards

    This is how DeepSeek-R1 achieved state-of-the-art reasoning!

    Resources:
    - TRL docs: https://huggingface.co/docs/trl
    - DeepSeek-R1 paper: https://arxiv.org/abs/2501.12948
    - GRPO paper: https://arxiv.org/abs/2402.03300
    """)


if __name__ == "__main__":
    main()
