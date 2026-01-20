#!/usr/bin/env python3
"""
RLVR & GRPO: Reinforcement Learning with Verifiable Rewards
============================================================

This tutorial covers the techniques used to train reasoning models like
DeepSeek-R1, which achieved remarkable performance on math and coding tasks.

Key Concepts:
1. RLVR (RL with Verifiable Rewards) - Using programmatic verification instead of reward models
2. GRPO (Group Relative Policy Optimization) - A simpler, more efficient alternative to PPO
3. Process Reward Models (PRM) - Rewarding intermediate reasoning steps

Why RLVR?
---------
Traditional RLHF uses a learned reward model to score responses. But for tasks
like math and coding, we can VERIFY correctness directly:
- Math: Check if the answer is correct
- Code: Run tests and check if they pass
- Logic: Verify logical consistency

This gives us:
✓ Perfect reward signals (no reward model errors)
✓ No reward hacking (can't fool a verifier)
✓ Scalable (no human labeling needed)

Prerequisites:
    pip install torch transformers datasets trl

References:
- DeepSeek-R1 Paper: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL"
- GRPO Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import random
import re
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: UNDERSTANDING RLVR
# =============================================================================

def explain_rlvr():
    """
    Explain RLVR and why it's powerful for reasoning tasks.
    """
    explanation = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║     RLVR: REINFORCEMENT LEARNING WITH VERIFIABLE REWARDS                  ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    THE KEY INSIGHT:
    ────────────────
    For tasks with objectively correct answers, we don't need a learned reward
    model - we can VERIFY correctness directly!

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    RLHF vs RLVR Comparison                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   RLHF (Traditional):                                                    │
    │   ┌──────────┐      ┌──────────────┐      ┌──────────┐                  │
    │   │  Policy  │ ──►  │ Reward Model │ ──►  │  Reward  │                  │
    │   │  Output  │      │  (Learned)   │      │  Score   │                  │
    │   └──────────┘      └──────────────┘      └──────────┘                  │
    │                            ↑                                             │
    │                     Can be fooled!                                       │
    │                     Can have errors!                                     │
    │                                                                          │
    │   RLVR (Verifiable):                                                     │
    │   ┌──────────┐      ┌──────────────┐      ┌──────────┐                  │
    │   │  Policy  │ ──►  │   Verifier   │ ──►  │  Reward  │                  │
    │   │  Output  │      │ (Programmatic)│      │  (0 or 1)│                  │
    │   └──────────┘      └──────────────┘      └──────────┘                  │
    │                            ↑                                             │
    │                     Perfect signal!                                      │
    │                     Cannot be fooled!                                    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    VERIFIABLE TASKS:
    ─────────────────

    ┌─────────────────┬────────────────────────────────────────┐
    │ Task            │ Verification Method                    │
    ├─────────────────┼────────────────────────────────────────┤
    │ Math Problems   │ Compare final answer to ground truth   │
    │ Code Generation │ Run unit tests                         │
    │ Logic Puzzles   │ Check logical consistency              │
    │ Theorem Proving │ Verify proof steps                     │
    │ SQL Queries     │ Execute and compare results            │
    │ Regex           │ Test against examples                  │
    └─────────────────┴────────────────────────────────────────┘

    THE DEEPSEEK APPROACH:
    ──────────────────────

    DeepSeek-R1 used RLVR to train reasoning capabilities:

    1. Start with a base model (DeepSeek-V3)
    2. Generate multiple solutions per problem (sampling)
    3. Verify which solutions are correct
    4. Use GRPO to reinforce correct reasoning patterns
    5. The model learns to "think step by step" naturally!

    Key finding: The model spontaneously developed:
    - Chain-of-thought reasoning
    - Self-verification ("let me check...")
    - Backtracking ("wait, that's wrong...")

    All from just rewarding correct final answers!

    ═══════════════════════════════════════════════════════════════════════════
    """
    print(explanation)


# =============================================================================
# PART 2: UNDERSTANDING GRPO
# =============================================================================

def explain_grpo():
    """
    Explain GRPO (Group Relative Policy Optimization).
    """
    explanation = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║     GRPO: GROUP RELATIVE POLICY OPTIMIZATION                              ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    GRPO is a simpler alternative to PPO that works especially well with
    verifiable rewards. It was introduced by DeepSeek for math reasoning.

    THE KEY IDEA:
    ─────────────
    Instead of using a value function (like PPO), GRPO computes advantages
    by comparing responses WITHIN A GROUP sampled for the same prompt.

    PPO Advantage:    A = R - V(s)         ← Needs a value network
    GRPO Advantage:   A = R - mean(R_group) ← Just uses group statistics!

    THE ALGORITHM:
    ──────────────

    For each prompt:
    1. Sample N responses from the policy
    2. Compute reward for each response (using verifier)
    3. Compute advantage: A_i = R_i - mean(R_1, ..., R_N)
    4. Update policy to increase probability of high-advantage responses

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         GRPO Visualization                               │
    │                                                                          │
    │   Prompt: "What is 15 + 27?"                                            │
    │                                                                          │
    │   Sample N=4 responses:                                                  │
    │   ┌────────────────────────┬────────┬───────────┬───────────┐           │
    │   │ Response               │ Correct│ Reward    │ Advantage │           │
    │   ├────────────────────────┼────────┼───────────┼───────────┤           │
    │   │ "15+27=42"             │   ✓    │    1.0    │   +0.5    │ ← boost   │
    │   │ "Let me add: 42"       │   ✓    │    1.0    │   +0.5    │ ← boost   │
    │   │ "15+27=52"             │   ✗    │    0.0    │   -0.5    │ ← reduce  │
    │   │ "The answer is 32"     │   ✗    │    0.0    │   -0.5    │ ← reduce  │
    │   └────────────────────────┴────────┴───────────┴───────────┘           │
    │                                                                          │
    │   Mean reward = 0.5                                                      │
    │   Advantage = Reward - Mean                                              │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    THE GRPO LOSS:
    ──────────────

    L_GRPO = -E[ min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A) ] + β * KL

    Where:
    - r(θ) = π_θ(a|s) / π_old(a|s)  (probability ratio, like PPO)
    - A = reward - group_mean        (group-relative advantage)
    - β * KL = KL penalty to reference policy

    WHY GRPO WORKS WELL:
    ────────────────────

    ✓ No value network needed (simpler, fewer parameters)
    ✓ Natural baseline from group (reduces variance)
    ✓ Works great with binary rewards (correct/incorrect)
    ✓ Efficient - one forward pass for advantage computation

    GRPO vs PPO:
    ────────────

    ┌─────────────────┬──────────────────┬──────────────────┐
    │ Aspect          │ PPO              │ GRPO             │
    ├─────────────────┼──────────────────┼──────────────────┤
    │ Value Network   │ Required         │ Not needed       │
    │ Advantage       │ GAE (complex)    │ Group mean       │
    │ Memory          │ Higher           │ Lower            │
    │ Implementation  │ Complex          │ Simpler          │
    │ Best for        │ Dense rewards    │ Sparse/binary    │
    └─────────────────┴──────────────────┴──────────────────┘

    ═══════════════════════════════════════════════════════════════════════════
    """
    print(explanation)


# =============================================================================
# PART 3: IMPLEMENTING VERIFIERS
# =============================================================================

class MathVerifier:
    """
    Verifier for mathematical problems.

    Extracts the final numerical answer and compares to ground truth.
    """

    def __init__(self):
        # Patterns to extract numerical answers
        self.answer_patterns = [
            r"(?:answer|result|equals|=)\s*[:\s]*(-?\d+\.?\d*)",
            r"(-?\d+\.?\d*)\s*$",  # Number at end
            r"\\boxed{(-?\d+\.?\d*)}",  # LaTeX boxed
            r"####\s*(-?\d+\.?\d*)",  # Common format
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

    def verify(self, response: str, ground_truth: float, tolerance: float = 1e-6) -> Tuple[bool, float]:
        """
        Verify if the response contains the correct answer.

        Returns:
            (is_correct, reward)
        """
        extracted = self.extract_answer(response)

        if extracted is None:
            return False, 0.0

        is_correct = abs(extracted - ground_truth) < tolerance
        reward = 1.0 if is_correct else 0.0

        return is_correct, reward


class CodeVerifier:
    """
    Verifier for code generation tasks.

    Executes code and runs test cases.
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from response."""
        # Look for code blocks
        code_patterns = [
            r"```python\n(.*?)```",
            r"```\n(.*?)```",
            r"def \w+\(.*?\):.*?(?=\n\n|\Z)",
        ]

        for pattern in code_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        return None

    def verify(self, response: str, test_cases: List[Dict[str, Any]]) -> Tuple[bool, float]:
        """
        Verify if the code passes all test cases.

        Args:
            response: The model's response containing code
            test_cases: List of {"input": ..., "expected": ...}

        Returns:
            (all_passed, reward)
        """
        code = self.extract_code(response)
        if code is None:
            return False, 0.0

        passed = 0
        total = len(test_cases)

        for test in test_cases:
            try:
                # Create a safe execution environment
                local_env = {}
                exec(code, {"__builtins__": {}}, local_env)

                # Find the function
                func = None
                for name, obj in local_env.items():
                    if callable(obj):
                        func = obj
                        break

                if func is None:
                    continue

                # Run test
                result = func(test["input"])
                if result == test["expected"]:
                    passed += 1

            except Exception:
                continue

        reward = passed / total if total > 0 else 0.0
        all_passed = passed == total

        return all_passed, reward


class ProcessRewardModel:
    """
    Process Reward Model (PRM) - rewards intermediate reasoning steps.

    Instead of only rewarding the final answer, PRM rewards each step
    of the reasoning process. This provides denser feedback.
    """

    def __init__(self):
        # Keywords indicating good reasoning
        self.positive_markers = [
            "let me", "first", "then", "therefore", "because",
            "step 1", "step 2", "this means", "we can see",
            "checking", "verify", "makes sense"
        ]

        # Keywords indicating potential issues
        self.negative_markers = [
            "i'm not sure", "maybe", "i think", "probably",
            "skip", "ignore"
        ]

    def score_reasoning(self, response: str) -> float:
        """
        Score the quality of reasoning steps.

        This is a simplified heuristic. In practice, you'd train a
        neural network to score reasoning quality.
        """
        response_lower = response.lower()
        score = 0.0

        # Reward for showing work
        steps = response.split('\n')
        if len(steps) > 3:
            score += 0.2  # Multiple steps

        # Reward for reasoning markers
        for marker in self.positive_markers:
            if marker in response_lower:
                score += 0.1

        # Penalty for uncertainty markers
        for marker in self.negative_markers:
            if marker in response_lower:
                score -= 0.05

        return max(0.0, min(1.0, score))

    def compute_reward(
        self,
        response: str,
        is_correct: bool,
        outcome_weight: float = 0.7,
        process_weight: float = 0.3
    ) -> float:
        """
        Combine outcome reward with process reward.

        Args:
            response: The model's response
            is_correct: Whether the final answer is correct
            outcome_weight: Weight for correct/incorrect (default 0.7)
            process_weight: Weight for reasoning quality (default 0.3)

        Returns:
            Combined reward in [0, 1]
        """
        outcome_reward = 1.0 if is_correct else 0.0
        process_reward = self.score_reasoning(response)

        return outcome_weight * outcome_reward + process_weight * process_reward


# =============================================================================
# PART 4: GRPO IMPLEMENTATION
# =============================================================================

@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Sampling
    group_size: int = 4  # Number of responses per prompt
    max_new_tokens: int = 256
    temperature: float = 0.7

    # Training
    learning_rate: float = 1e-5
    clip_epsilon: float = 0.2  # PPO-style clipping
    kl_coef: float = 0.1  # KL penalty coefficient

    # Optimization
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) Trainer.

    Implements the GRPO algorithm for training with verifiable rewards.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer,
        verifier: Callable,
        config: GRPOConfig,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.tokenizer = tokenizer
        self.verifier = verifier
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # Statistics
        self.stats = {
            "rewards": [],
            "advantages": [],
            "kl_divergence": [],
            "loss": [],
        }

    def generate_group(self, prompt: str) -> List[str]:
        """
        Generate a group of responses for the same prompt.
        """
        self.model.eval()
        responses = []

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            for _ in range(self.config.group_size):
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                # Remove the prompt from the response
                response = response[len(prompt):].strip()
                responses.append(response)

        self.model.train()
        return responses

    def compute_group_advantages(
        self,
        responses: List[str],
        ground_truth: Any
    ) -> Tuple[List[float], List[float]]:
        """
        Compute rewards and advantages for a group of responses.

        Advantage = reward - group_mean (the key GRPO insight!)
        """
        rewards = []

        for response in responses:
            is_correct, reward = self.verifier(response, ground_truth)
            rewards.append(reward)

        # Group-relative advantage
        mean_reward = sum(rewards) / len(rewards)
        advantages = [r - mean_reward for r in rewards]

        # Normalize advantages
        std = (sum(a**2 for a in advantages) / len(advantages)) ** 0.5
        if std > 0:
            advantages = [a / (std + 1e-8) for a in advantages]

        return rewards, advantages

    def compute_log_probs(
        self,
        prompt: str,
        response: str,
        model: nn.Module
    ) -> torch.Tensor:
        """
        Compute log probability of response given prompt.
        """
        full_text = prompt + response
        input_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
        prompt_len = len(self.tokenizer.encode(prompt))

        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(input_ids, labels=input_ids)
            logits = outputs.logits

        # Get log probs for response tokens only
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs of actual tokens
        response_log_probs = log_probs[0, prompt_len-1:-1, :]
        response_tokens = input_ids[0, prompt_len:]

        token_log_probs = response_log_probs.gather(1, response_tokens.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.sum()

    def compute_kl_divergence(
        self,
        prompt: str,
        response: str
    ) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference.
        """
        full_text = prompt + response
        input_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids)
            ref_logits = ref_outputs.logits

        policy_outputs = self.model(input_ids)
        policy_logits = policy_outputs.logits

        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)

        # KL(policy || ref)
        kl = (torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)).sum(-1).mean()

        return kl

    def grpo_step(
        self,
        prompt: str,
        responses: List[str],
        advantages: List[float]
    ) -> Dict[str, float]:
        """
        Perform one GRPO optimization step.
        """
        total_loss = 0.0
        total_kl = 0.0

        for response, advantage in zip(responses, advantages):
            # Compute log probabilities
            policy_log_prob = self.compute_log_probs(prompt, response, self.model)

            with torch.no_grad():
                ref_log_prob = self.compute_log_probs(prompt, response, self.ref_model)

            # Probability ratio
            ratio = torch.exp(policy_log_prob - ref_log_prob)

            # Clipped objective (like PPO)
            advantage_tensor = torch.tensor(advantage, device=self.device)
            surr1 = ratio * advantage_tensor
            surr2 = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon
            ) * advantage_tensor

            policy_loss = -torch.min(surr1, surr2)

            # KL penalty
            kl = self.compute_kl_divergence(prompt, response)

            # Total loss
            loss = policy_loss + self.config.kl_coef * kl

            total_loss += loss
            total_kl += kl.item()

        # Average loss
        total_loss = total_loss / len(responses)

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "kl": total_kl / len(responses),
        }

    def train_step(
        self,
        prompt: str,
        ground_truth: Any
    ) -> Dict[str, float]:
        """
        Complete GRPO training step for one prompt.

        1. Generate group of responses
        2. Verify and compute rewards
        3. Compute group-relative advantages
        4. Update policy with GRPO
        """
        # Step 1: Generate group
        responses = self.generate_group(prompt)

        # Step 2 & 3: Compute rewards and advantages
        rewards, advantages = self.compute_group_advantages(responses, ground_truth)

        # Step 4: GRPO update
        stats = self.grpo_step(prompt, responses, advantages)

        # Track statistics
        self.stats["rewards"].append(sum(rewards) / len(rewards))
        self.stats["advantages"].append(sum(abs(a) for a in advantages) / len(advantages))
        self.stats["kl_divergence"].append(stats["kl"])
        self.stats["loss"].append(stats["loss"])

        return {
            "mean_reward": sum(rewards) / len(rewards),
            "correct_ratio": sum(1 for r in rewards if r > 0.5) / len(rewards),
            **stats
        }


# =============================================================================
# PART 5: TRAINING EXAMPLE
# =============================================================================

def create_math_dataset():
    """
    Create a dataset of math problems with ground truth answers.
    """
    problems = [
        {"question": "What is 15 + 27?", "answer": 42},
        {"question": "Calculate 8 × 7", "answer": 56},
        {"question": "What is 100 - 37?", "answer": 63},
        {"question": "Compute 144 ÷ 12", "answer": 12},
        {"question": "What is 25 × 4?", "answer": 100},
        {"question": "Calculate 18 + 45 + 37", "answer": 100},
        {"question": "What is 200 - 75?", "answer": 125},
        {"question": "Compute 15 × 15", "answer": 225},
        {"question": "What is 1000 ÷ 8?", "answer": 125},
        {"question": "Calculate 33 + 67", "answer": 100},
        {"question": "What is 12 × 12?", "answer": 144},
        {"question": "Compute 500 - 123", "answer": 377},
    ]
    return problems


def run_grpo_training():
    """
    Run GRPO training on math problems.
    """
    print("\n" + "=" * 70)
    print("PART 5: GRPO TRAINING EXAMPLE")
    print("=" * 70)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import copy
    except ImportError:
        print("\nPlease install: pip install transformers")
        return

    # Use a small model
    model_name = "distilgpt2"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = copy.deepcopy(model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create verifier
    math_verifier = MathVerifier()

    def verifier_fn(response: str, ground_truth: float) -> Tuple[bool, float]:
        return math_verifier.verify(response, ground_truth)

    # Create trainer
    config = GRPOConfig(
        group_size=4,
        max_new_tokens=50,
        temperature=0.8,
        learning_rate=1e-5,
        kl_coef=0.1,
    )

    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        verifier=verifier_fn,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create dataset
    dataset = create_math_dataset()

    print("\n" + "-" * 50)
    print("Training Configuration:")
    print("-" * 50)
    print(f"  Group size: {config.group_size}")
    print(f"  Temperature: {config.temperature}")
    print(f"  KL coefficient: {config.kl_coef}")
    print(f"  Dataset size: {len(dataset)}")

    # Training loop
    print("\n" + "-" * 50)
    print("Starting GRPO Training")
    print("-" * 50)
    print("\nThe algorithm will:")
    print("1. Sample multiple responses per problem")
    print("2. Verify which ones are correct")
    print("3. Compute group-relative advantages")
    print("4. Update policy to favor correct responses\n")

    num_epochs = 2

    for epoch in range(num_epochs):
        print(f"\n{'='*20} Epoch {epoch + 1}/{num_epochs} {'='*20}")

        epoch_rewards = []
        epoch_correct = []

        for i, problem in enumerate(dataset):
            prompt = f"Solve this math problem and give the numerical answer:\n{problem['question']}\nAnswer:"

            stats = trainer.train_step(prompt, problem['answer'])

            epoch_rewards.append(stats['mean_reward'])
            epoch_correct.append(stats['correct_ratio'])

            if (i + 1) % 4 == 0:
                print(f"\n  Problem {i+1}/{len(dataset)}:")
                print(f"    Question: {problem['question']}")
                print(f"    Correct answer: {problem['answer']}")
                print(f"    Correct ratio: {stats['correct_ratio']:.1%}")
                print(f"    Mean reward: {stats['mean_reward']:.3f}")
                print(f"    KL divergence: {stats['kl']:.4f}")

        print(f"\n  Epoch {epoch + 1} Summary:")
        print(f"    Mean reward: {sum(epoch_rewards)/len(epoch_rewards):.3f}")
        print(f"    Correct ratio: {sum(epoch_correct)/len(epoch_correct):.1%}")

    print("\n" + "=" * 50)
    print("GRPO Training Complete!")
    print("=" * 50)

    # Test the trained model
    print("\n" + "-" * 50)
    print("Testing trained model")
    print("-" * 50)

    test_problems = [
        {"question": "What is 23 + 19?", "answer": 42},
        {"question": "Calculate 7 × 8", "answer": 56},
    ]

    model.eval()
    for problem in test_problems:
        prompt = f"Solve: {problem['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        is_correct, _ = math_verifier.verify(response, problem['answer'])

        print(f"\n  Q: {problem['question']}")
        print(f"  Model: {response}")
        print(f"  Correct: {'✓' if is_correct else '✗'} (expected {problem['answer']})")


# =============================================================================
# PART 6: COMPARISON WITH OTHER METHODS
# =============================================================================

def show_comparison():
    """
    Compare RLVR/GRPO with other methods.
    """
    comparison = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              COMPARISON: RLVR/GRPO vs OTHER METHODS                       ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
    │ Method          │ Reward Source   │ Best For        │ Key Advantage   │
    ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
    │ RLHF + PPO      │ Learned RM      │ General tasks   │ Flexible        │
    │                 │                 │ Subjective      │                 │
    ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
    │ DPO             │ Preferences     │ General align.  │ Simple          │
    │                 │ (implicit)      │ No RL needed    │                 │
    ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
    │ RLVR + GRPO     │ Verifier        │ Math, Code      │ Perfect signal  │
    │                 │ (programmatic)  │ Reasoning       │ No RM errors    │
    ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
    │ RLVR + PRM      │ Process reward  │ Multi-step      │ Dense feedback  │
    │                 │                 │ reasoning       │ Better learning │
    └─────────────────┴─────────────────┴─────────────────┴─────────────────┘

    WHEN TO USE RLVR/GRPO:
    ──────────────────────

    ✓ You have objectively correct answers (math, code, logic)
    ✓ You can programmatically verify correctness
    ✓ You want to avoid reward model errors
    ✓ You're training reasoning capabilities

    WHEN TO USE RLHF/DPO:
    ─────────────────────

    ✓ Tasks are subjective (writing quality, helpfulness)
    ✓ No clear "correct" answer exists
    ✓ You have human preference data
    ✓ General-purpose alignment

    THE DEEPSEEK RECIPE:
    ────────────────────

    DeepSeek-R1 combined multiple techniques:

    1. Start with strong base model (DeepSeek-V3)
    2. Cold-start with SFT on reasoning examples
    3. RLVR with GRPO on math/code (verifiable)
    4. RL with reward model on general tasks
    5. Final SFT to clean up formatting

    Key insight: Different rewards for different tasks!

    ═══════════════════════════════════════════════════════════════════════════
    """
    print(comparison)


# =============================================================================
# PART 7: KEY TAKEAWAYS
# =============================================================================

def show_takeaways():
    """
    Summarize the key takeaways.
    """
    takeaways = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                         KEY TAKEAWAYS                                     ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    1. RLVR (Verifiable Rewards):
       - Use programmatic verification instead of learned reward models
       - Perfect for math, code, logic, and other verifiable tasks
       - No reward hacking possible - verifier can't be fooled

    2. GRPO (Group Relative Policy Optimization):
       - Simpler than PPO - no value network needed
       - Advantage = reward - group_mean
       - Works great with binary (correct/incorrect) rewards
       - The algorithm behind DeepSeek-R1's success

    3. Process Reward Models (PRM):
       - Reward intermediate reasoning steps, not just final answer
       - Provides denser feedback for learning
       - Helps model learn HOW to think, not just WHAT to answer

    4. Emergent Behaviors:
       - Models trained with RLVR spontaneously develop:
         • Chain-of-thought reasoning
         • Self-verification
         • Backtracking on errors
       - All from just rewarding correct answers!

    5. Practical Tips:
       - Use larger group sizes (8-16) for more stable training
       - Balance outcome reward with process reward
       - Start with a good SFT model before RLVR
       - Mix verifiable and general tasks for well-rounded models

    RESOURCES:
    ──────────
    - DeepSeek-R1 Paper: https://arxiv.org/abs/2501.12948
    - DeepSeekMath Paper (GRPO): https://arxiv.org/abs/2402.03300
    - Let's Verify Step by Step (PRM): https://arxiv.org/abs/2305.20050

    ═══════════════════════════════════════════════════════════════════════════
    """
    print(takeaways)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the complete RLVR/GRPO tutorial."""
    print("=" * 70)
    print("   RLVR & GRPO: TRAINING REASONING MODELS")
    print("   The Techniques Behind DeepSeek-R1")
    print("=" * 70)

    # Part 1: Why RLVR?
    explain_rlvr()
    input("\nPress Enter to continue to Part 2 (GRPO Algorithm)...")

    # Part 2: GRPO Explained
    explain_grpo()
    input("\nPress Enter to continue to Part 3 (Verifiers)...")

    # Part 3: Verifiers (shown inline)
    print("\n" + "=" * 70)
    print("PART 3: IMPLEMENTING VERIFIERS")
    print("=" * 70)
    print("""
    We've implemented several verifiers:

    1. MathVerifier - Extracts and verifies numerical answers
    2. CodeVerifier - Runs code against test cases
    3. ProcessRewardModel - Scores reasoning quality

    Example usage:

    >>> verifier = MathVerifier()
    >>> verifier.verify("The answer is 42", ground_truth=42)
    (True, 1.0)

    >>> verifier.verify("I think it's 50", ground_truth=42)
    (False, 0.0)
    """)

    # Demo the verifier
    verifier = MathVerifier()
    print("\n  Live demo:")
    test_cases = [
        ("Let me calculate: 15 + 27 = 42", 42),
        ("The result is 100", 42),
        ("Step 1: Add the numbers. Answer: 42", 42),
    ]
    for response, gt in test_cases:
        is_correct, reward = verifier.verify(response, gt)
        status = "✓" if is_correct else "✗"
        print(f"    '{response}' → {status} (reward: {reward})")

    input("\nPress Enter to continue to Part 4 (GRPO Implementation)...")

    # Part 4: GRPO Implementation (shown inline)
    print("\n" + "=" * 70)
    print("PART 4: GRPO IMPLEMENTATION")
    print("=" * 70)
    print("""
    The GRPOTrainer class implements:

    1. generate_group(): Sample N responses per prompt
    2. compute_group_advantages(): Advantage = reward - mean
    3. grpo_step(): Update policy with clipped objective + KL penalty

    Key difference from PPO:
    - No value network
    - Advantage is relative to group mean
    - Much simpler implementation
    """)

    input("\nPress Enter to continue to Part 5 (Training Example)...")

    # Part 5: Training
    run_grpo_training()
    input("\nPress Enter to continue to Part 6 (Comparison)...")

    # Part 6: Comparison
    show_comparison()

    # Part 7: Takeaways
    show_takeaways()

    print("\n" + "=" * 70)
    print("Tutorial Complete!")
    print("=" * 70)
    print("""
    You've learned:
    ✓ Why RLVR works for reasoning tasks
    ✓ How GRPO simplifies RL training
    ✓ How to implement verifiers
    ✓ The techniques behind DeepSeek-R1

    Next steps:
    1. Try GRPO on coding problems with test cases
    2. Implement a Process Reward Model
    3. Combine RLVR with general RLHF
    4. Scale up to larger models
    """)


if __name__ == "__main__":
    main()
