#!/usr/bin/env python3
"""
Reinforcement Learning for LLM Training - Comprehensive Tutorial
=================================================================

This tutorial provides a deep dive into how Reinforcement Learning (RL) is used
to train and align Large Language Models. We'll cover:

1. WHY use RL for LLMs? (The alignment problem)
2. The RLHF pipeline (Reward Model → PPO)
3. Hands-on PPO training with TRL
4. Modern alternatives (DPO, ORPO) - simpler and often better
5. Practical comparisons and when to use what

This is the technique that transformed GPT-3 into ChatGPT!

Prerequisites:
    pip install trl transformers datasets torch

Note: This tutorial is designed to actually run and train models,
not just show code snippets.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: WHY REINFORCEMENT LEARNING FOR LLMS?
# =============================================================================

def explain_why_rl():
    """
    Explain why we need RL for LLMs.
    """
    explanation = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              WHY USE REINFORCEMENT LEARNING FOR LLMS?                     ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    THE PROBLEM: Pre-trained LLMs are not aligned with human preferences
    ────────────────────────────────────────────────────────────────────

    A pre-trained LLM (like GPT-3) learns to predict the next token from internet
    text. This means it learns to mimic ALL of the internet - including:

    ✗ Toxic content
    ✗ Misinformation
    ✗ Unhelpful responses
    ✗ Harmful instructions

    Just doing supervised fine-tuning (SFT) on good examples helps, but the model
    can still produce problematic outputs because it doesn't truly understand
    what makes a response "good" vs "bad".

    THE SOLUTION: Teach the model what humans prefer using RL
    ─────────────────────────────────────────────────────────

    Reinforcement Learning allows us to:

    ✓ Define a reward signal based on human preferences
    ✓ Optimize the model to maximize this reward
    ✓ Balance between being helpful AND being safe
    ✓ Teach nuanced preferences that are hard to specify with rules

    THE RLHF PIPELINE:
    ──────────────────

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Step 1    │     │   Step 2    │     │   Step 3    │
    │             │ ──► │             │ ──► │             │
    │     SFT     │     │   Reward    │     │     PPO     │
    │  (Imitate)  │     │   Model     │     │ (Optimize)  │
    └─────────────┘     └─────────────┘     └─────────────┘

    Train model to      Learn what         Use RL to make
    follow format       humans prefer      model maximize
    from examples       from comparisons   the reward

    WHAT MAKES RL SPECIAL FOR THIS?
    ───────────────────────────────

    1. REWARD SHAPING: We can combine multiple objectives
       - Helpfulness reward
       - Safety reward
       - Factuality reward
       - Style reward

    2. KL PENALTY: Prevents model from "gaming" the reward
       - Keeps model close to original behavior
       - Prevents degenerate outputs

    3. ONLINE LEARNING: Model generates its own training data
       - Explores the response space
       - Learns from its own mistakes

    ═══════════════════════════════════════════════════════════════════════════
    """
    print(explanation)


# =============================================================================
# PART 2: THE CORE CONCEPTS
# =============================================================================

def explain_core_concepts():
    """
    Explain the core RL concepts as applied to LLMs.
    """
    concepts = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    RL CONCEPTS APPLIED TO LLMS                            ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    In RL terminology:
    ──────────────────

    ┌──────────────────┬──────────────────────────────────────────────────────┐
    │ RL Concept       │ LLM Equivalent                                       │
    ├──────────────────┼──────────────────────────────────────────────────────┤
    │ Agent            │ The language model                                   │
    │ Environment      │ The conversation/task                                │
    │ State            │ The prompt + conversation history                    │
    │ Action           │ Generating the next token (or full response)        │
    │ Policy           │ The model's probability distribution over tokens    │
    │ Reward           │ Score from reward model (human preference proxy)    │
    │ Episode          │ One complete prompt → response interaction          │
    └──────────────────┴──────────────────────────────────────────────────────┘

    The PPO Objective (simplified):
    ───────────────────────────────

    maximize: E[reward(response)] - β * KL(policy || reference_policy)
              ─────────────────────   ─────────────────────────────────
              Get high rewards        Don't change too much from
              from reward model       the original model

    Why KL penalty?
    ───────────────

    Without KL penalty, the model might find "reward hacks":
    - Repeating phrases the reward model likes
    - Generating gibberish that scores high
    - Losing language ability while maximizing reward

    The KL term keeps the model "sane" by penalizing deviation
    from the original pre-trained behavior.

    ═══════════════════════════════════════════════════════════════════════════
    """
    print(concepts)


# =============================================================================
# PART 3: HANDS-ON PPO TRAINING
# =============================================================================

def run_ppo_training():
    """
    Actually run PPO training on a small model.
    """
    print("\n" + "=" * 70)
    print("PART 3: HANDS-ON PPO TRAINING")
    print("=" * 70)

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
        from datasets import Dataset
    except ImportError:
        print("\nPlease install: pip install trl transformers datasets")
        return

    # Use a tiny model for demonstration
    model_name = "distilgpt2"
    print(f"\nLoading model: {model_name}")
    print("(Using a small model so this runs quickly on CPU)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load policy model with value head (required for PPO)
    # The value head predicts expected future rewards
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    # Load reference model (frozen copy for KL penalty)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    print(f"Policy model parameters: {sum(p.numel() for p in policy_model.parameters()):,}")

    # For this demo, we'll use a simple reward function
    # In practice, you'd use a trained reward model
    print("\n" + "-" * 50)
    print("Creating a simple reward function")
    print("-" * 50)

    def simple_reward_function(responses: List[str]) -> List[float]:
        """
        A simple reward function for demonstration.

        In practice, this would be a neural network trained on human preferences.
        Here we use simple heuristics:
        - Reward longer, more detailed responses
        - Penalize very short responses
        - Penalize repetition
        """
        rewards = []
        for response in responses:
            reward = 0.0

            # Reward based on length (but not too long)
            word_count = len(response.split())
            if word_count < 5:
                reward -= 1.0  # Too short
            elif word_count < 20:
                reward += 0.5  # Good length
            elif word_count < 50:
                reward += 1.0  # Great length
            else:
                reward += 0.5  # Maybe too verbose

            # Penalize repetition
            words = response.lower().split()
            unique_ratio = len(set(words)) / max(len(words), 1)
            reward += unique_ratio - 0.5  # Bonus for variety

            # Reward for ending with punctuation (complete sentences)
            if response.strip().endswith(('.', '!', '?')):
                reward += 0.3

            rewards.append(reward)

        return rewards

    # Test the reward function
    test_responses = [
        "Yes.",  # Too short
        "Python is a programming language that is widely used.",  # Good
        "the the the the the",  # Repetitive
    ]
    test_rewards = simple_reward_function(test_responses)
    print("\nReward function test:")
    for resp, rew in zip(test_responses, test_rewards):
        print(f"  '{resp[:40]}...' → reward: {rew:.2f}")

    # PPO Configuration
    print("\n" + "-" * 50)
    print("PPO Configuration")
    print("-" * 50)

    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=4,           # Number of samples per batch
        mini_batch_size=2,      # For gradient accumulation
        ppo_epochs=2,           # PPO optimization epochs per batch
        kl_penalty="kl",        # Type of KL penalty
        target_kl=0.1,          # Target KL divergence
        init_kl_coef=0.2,       # Initial KL penalty coefficient
        log_with=None,          # Disable logging for demo
    )

    print(f"  Learning rate: {ppo_config.learning_rate}")
    print(f"  Batch size: {ppo_config.batch_size}")
    print(f"  PPO epochs: {ppo_config.ppo_epochs}")
    print(f"  Target KL: {ppo_config.target_kl}")

    # Create prompts
    prompts = [
        "Explain what machine learning is:",
        "What is the capital of France?",
        "How do computers work?",
        "What is Python programming?",
        "Describe the solar system:",
        "What is artificial intelligence?",
        "How does the internet work?",
        "What is a neural network?",
    ]

    print(f"\n  Training prompts: {len(prompts)}")

    # Create PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        processing_class=tokenizer,
    )

    # Training loop
    print("\n" + "-" * 50)
    print("Running PPO Training Loop")
    print("-" * 50)
    print("\nThis demonstrates the core RLHF training loop:")
    print("1. Generate responses with current policy")
    print("2. Score responses with reward function")
    print("3. Update policy using PPO")
    print()

    num_epochs = 2

    for epoch in range(num_epochs):
        print(f"\n{'='*20} Epoch {epoch + 1}/{num_epochs} {'='*20}")

        epoch_rewards = []

        # Process prompts in batches
        for i in range(0, len(prompts), ppo_config.batch_size):
            batch_prompts = prompts[i:i + ppo_config.batch_size]

            # Tokenize prompts
            query_tensors = [
                tokenizer.encode(prompt, return_tensors="pt").squeeze()
                for prompt in batch_prompts
            ]

            # Step 1: Generate responses
            response_tensors = []
            for query in query_tensors:
                response = ppo_trainer.generate(
                    query.unsqueeze(0),
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response_tensors.append(response.squeeze())

            # Decode responses for reward computation
            responses = [
                tokenizer.decode(r[len(q):], skip_special_tokens=True)
                for q, r in zip(query_tensors, response_tensors)
            ]

            # Step 2: Compute rewards
            rewards = simple_reward_function(responses)
            reward_tensors = [torch.tensor([r]) for r in rewards]

            # Step 3: PPO update
            stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

            # Track statistics
            epoch_rewards.extend(rewards)

            # Print batch info
            print(f"\n  Batch {i//ppo_config.batch_size + 1}:")
            print(f"    Mean reward: {sum(rewards)/len(rewards):.3f}")
            if 'ppo/policy_loss' in stats:
                print(f"    Policy loss: {stats['ppo/policy_loss']:.4f}")
            if 'ppo/kl' in stats:
                print(f"    KL divergence: {stats['ppo/kl']:.4f}")

            # Show a sample
            print(f"    Sample: '{batch_prompts[0][:30]}...'")
            print(f"    Response: '{responses[0][:50]}...'")
            print(f"    Reward: {rewards[0]:.2f}")

        print(f"\n  Epoch {epoch + 1} mean reward: {sum(epoch_rewards)/len(epoch_rewards):.3f}")

    print("\n" + "=" * 50)
    print("PPO Training Complete!")
    print("=" * 50)

    # Test the trained model
    print("\n" + "-" * 50)
    print("Testing trained model")
    print("-" * 50)

    test_prompt = "What is artificial intelligence?"
    inputs = tokenizer(test_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = policy_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {test_prompt}")
    print(f"Response: {generated}")


# =============================================================================
# PART 4: DPO - THE SIMPLER ALTERNATIVE
# =============================================================================

def run_dpo_training():
    """
    Run DPO training - a simpler alternative to PPO.
    """
    print("\n" + "=" * 70)
    print("PART 4: DPO (Direct Preference Optimization)")
    print("=" * 70)

    dpo_explanation = """
    DPO: A Simpler Way to Learn from Preferences
    ─────────────────────────────────────────────

    The key insight of DPO:

    You don't need a separate reward model!

    Instead of:
    1. Train reward model on preferences
    2. Use PPO to maximize reward

    DPO directly optimizes the policy from preference pairs:

    Loss = -log σ(β * (log π(chosen) - log π(rejected)))

    Where:
    - π(chosen) = probability of preferred response
    - π(rejected) = probability of less-preferred response
    - β = temperature parameter
    - σ = sigmoid function

    This loss:
    - Increases probability of chosen response
    - Decreases probability of rejected response
    - Does so in a way that implicitly optimizes the same objective as PPO!

    Why DPO is often better:
    ────────────────────────

    ✓ No reward model needed (saves compute and complexity)
    ✓ No RL instabilities (it's just supervised learning!)
    ✓ Often achieves same or better results
    ✓ Much easier to implement and debug
    """
    print(dpo_explanation)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import DPOConfig, DPOTrainer
        from datasets import Dataset
    except ImportError:
        print("\nPlease install: pip install trl transformers datasets")
        return

    model_name = "distilgpt2"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # DPO needs preference pairs
    # Format: (prompt, chosen_response, rejected_response)
    preference_data = [
        {
            "prompt": "What is Python?",
            "chosen": "Python is a versatile, high-level programming language created by Guido van Rossum. It emphasizes code readability with its clean syntax and is widely used in web development, data science, AI, and automation.",
            "rejected": "its a snake lol",
        },
        {
            "prompt": "Explain machine learning.",
            "chosen": "Machine learning is a branch of artificial intelligence where systems learn patterns from data rather than being explicitly programmed. It enables computers to improve their performance on tasks through experience.",
            "rejected": "computers learn stuff i guess",
        },
        {
            "prompt": "How does the internet work?",
            "chosen": "The internet is a global network of connected computers that communicate using standardized protocols (TCP/IP). Data travels in packets through routers and switches, allowing devices worldwide to share information.",
            "rejected": "wifi magic",
        },
        {
            "prompt": "What is a database?",
            "chosen": "A database is an organized collection of structured data stored electronically. It uses a database management system (DBMS) to efficiently store, retrieve, and manage information for applications.",
            "rejected": "where data goes",
        },
        {
            "prompt": "Describe neural networks.",
            "chosen": "Neural networks are computing systems inspired by biological brains. They consist of layers of interconnected nodes (neurons) that process information, learn patterns from data, and make predictions.",
            "rejected": "brain computer thing",
        },
    ]

    # Duplicate for more training data
    preference_data = preference_data * 10
    dataset = Dataset.from_list(preference_data)

    print(f"\nPreference pairs: {len(preference_data)}")
    print("\nSample preference pair:")
    print(f"  Prompt: '{preference_data[0]['prompt']}'")
    print(f"  Chosen: '{preference_data[0]['chosen'][:60]}...'")
    print(f"  Rejected: '{preference_data[0]['rejected']}'")

    # DPO Config
    dpo_config = DPOConfig(
        output_dir="./dpo_output",
        beta=0.1,                          # KL penalty strength
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=5e-6,
        logging_steps=5,
        max_length=256,
        max_prompt_length=128,
        report_to="none",
        remove_unused_columns=False,
    )

    print(f"\nDPO Config:")
    print(f"  Beta (KL strength): {dpo_config.beta}")
    print(f"  Learning rate: {dpo_config.learning_rate}")
    print(f"  Epochs: {dpo_config.num_train_epochs}")

    # Create DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\n" + "-" * 50)
    print("Running DPO Training")
    print("-" * 50)
    print("\nTraining... (this may take a minute)")

    # Actually train!
    trainer.train()

    print("\n" + "=" * 50)
    print("DPO Training Complete!")
    print("=" * 50)

    # Test the trained model
    print("\n" + "-" * 50)
    print("Testing DPO-trained model")
    print("-" * 50)

    model.eval()
    test_prompts = [
        "What is Python?",
        "Explain artificial intelligence.",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {generated}")


# =============================================================================
# PART 5: COMPARISON AND RECOMMENDATIONS
# =============================================================================

def show_comparison():
    """
    Compare different RL methods for LLM training.
    """
    comparison = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              COMPARISON OF RL METHODS FOR LLMS                            ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    ┌─────────────┬─────────────────┬─────────────────┬─────────────────────────┐
    │ Method      │ Complexity      │ Performance     │ Best For                │
    ├─────────────┼─────────────────┼─────────────────┼─────────────────────────┤
    │ PPO (RLHF)  │ High            │ Good            │ Research, custom rewards│
    │             │ - Reward model  │                 │ Online learning         │
    │             │ - RL training   │                 │                         │
    │             │ - 4 models      │                 │                         │
    ├─────────────┼─────────────────┼─────────────────┼─────────────────────────┤
    │ DPO         │ Low             │ Great           │ Most use cases          │
    │             │ - Just SL       │                 │ Stability               │
    │             │ - 2 models      │                 │ Ease of use             │
    ├─────────────┼─────────────────┼─────────────────┼─────────────────────────┤
    │ ORPO        │ Very Low        │ Great           │ Memory-constrained      │
    │             │ - Single stage  │                 │ Simple pipelines        │
    │             │ - 1 model       │                 │                         │
    ├─────────────┼─────────────────┼─────────────────┼─────────────────────────┤
    │ KTO         │ Low             │ Good            │ When you only have      │
    │             │ - No pairs      │                 │ thumbs up/down data     │
    └─────────────┴─────────────────┴─────────────────┴─────────────────────────┘

    RECOMMENDATIONS:
    ────────────────

    🥇 Start with DPO
       - Simple, stable, works well
       - Easy to debug and iterate
       - Good baseline for any project

    🥈 Try ORPO if memory is limited
       - No reference model needed
       - Single training stage
       - State-of-the-art results

    🥉 Use PPO only when needed
       - Online learning scenarios
       - Custom reward functions
       - Research applications

    THE MODERN STACK:
    ─────────────────

    Pre-trained LLM (Llama, Mistral, etc.)
            │
            ▼
    SFT (Supervised Fine-Tuning)
    - Teach format and style
            │
            ▼
    DPO or ORPO (Preference Optimization)
    - Align with human preferences
            │
            ▼
    Evaluation & Iteration
    - Human evaluation
    - Benchmark testing

    ═══════════════════════════════════════════════════════════════════════════
    """
    print(comparison)


# =============================================================================
# PART 6: KEY TAKEAWAYS
# =============================================================================

def show_takeaways():
    """
    Summarize the key takeaways.
    """
    takeaways = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                         KEY TAKEAWAYS                                     ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    1. WHY RL FOR LLMS?
       - Pre-trained LLMs aren't aligned with human preferences
       - RL allows us to optimize for complex, nuanced objectives
       - The KL penalty prevents "reward hacking"

    2. THE CLASSIC PIPELINE (RLHF/PPO):
       - Train a reward model from human comparisons
       - Use PPO to optimize the policy to maximize reward
       - Works but is complex and can be unstable

    3. MODERN ALTERNATIVES:
       - DPO: Skip the reward model, optimize directly from preferences
       - ORPO: Even simpler, single-stage training
       - Both often work as well or better than PPO

    4. PRACTICAL ADVICE:
       - Start with DPO - it's simple and effective
       - Use high-quality preference data (quality > quantity)
       - The KL penalty (β) is crucial - tune it carefully
       - Always evaluate on held-out data

    5. THE BIG PICTURE:
       - RL is how we go from "language model" to "assistant"
       - It's the secret sauce behind ChatGPT, Claude, etc.
       - But the preference data quality matters most!

    RESOURCES:
    ──────────
    - TRL Documentation: https://huggingface.co/docs/trl
    - DPO Paper: "Direct Preference Optimization" (2023)
    - ORPO Paper: "ORPO: Monolithic Preference Optimization" (2024)
    - InstructGPT Paper: "Training language models to follow instructions" (2022)

    ═══════════════════════════════════════════════════════════════════════════
    """
    print(takeaways)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the complete RL for LLMs tutorial."""
    print("=" * 70)
    print("   REINFORCEMENT LEARNING FOR LLM TRAINING")
    print("   A Comprehensive Tutorial")
    print("=" * 70)

    # Part 1: Why RL?
    explain_why_rl()
    input("\nPress Enter to continue to Part 2 (Core Concepts)...")

    # Part 2: Core Concepts
    explain_core_concepts()
    input("\nPress Enter to continue to Part 3 (PPO Training)...")

    # Part 3: Hands-on PPO
    run_ppo_training()
    input("\nPress Enter to continue to Part 4 (DPO)...")

    # Part 4: DPO
    run_dpo_training()
    input("\nPress Enter to continue to Part 5 (Comparison)...")

    # Part 5: Comparison
    show_comparison()

    # Part 6: Takeaways
    show_takeaways()

    print("\n" + "=" * 70)
    print("Tutorial Complete!")
    print("=" * 70)
    print("""
    You've learned:
    ✓ Why RL is used for LLM alignment
    ✓ How PPO training works
    ✓ The simpler DPO alternative
    ✓ When to use which method

    Next steps:
    1. Try training on your own preference data
    2. Experiment with different β values
    3. Compare PPO vs DPO on your task
    4. Scale up to larger models with LoRA
    """)


if __name__ == "__main__":
    main()
