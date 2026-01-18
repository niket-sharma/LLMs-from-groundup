#!/usr/bin/env python3
"""
Reinforcement Learning for LLM Training Tutorial
=================================================

This example demonstrates how Reinforcement Learning from Human Feedback (RLHF)
is used to align language models with human preferences. This is the technique
used to train models like ChatGPT, Claude, and other instruction-following LLMs.

The RLHF Pipeline:
1. Pre-train a language model (we use our SmallGPT)
2. Train a reward model on human preference data
3. Fine-tune the LM using PPO to maximize the reward

This tutorial implements a simplified but complete version of this pipeline.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from tqdm import tqdm
import copy

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gpt import SmallGPT, create_small_gpt
from training.dataset import SimpleTokenizer
from utils.helpers import set_seed, get_device


# =============================================================================
# Part 1: Reward Model
# =============================================================================

class RewardModel(nn.Module):
    """
    Reward Model for RLHF.

    The reward model takes a sequence (prompt + response) and outputs a scalar
    reward indicating how "good" the response is according to human preferences.

    Architecture: GPT backbone (embeddings + transformer blocks) + linear head for scalar reward

    Note: We build this from the same components as SmallGPT but replace the
    language model head with a reward head that outputs a scalar.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Import the building blocks
        from models.embeddings import GPTEmbedding
        from models.feedforward import TransformerBlock

        # Embeddings (same as GPT)
        self.embedding = GPTEmbedding(vocab_size, d_model, max_seq_len, dropout)

        # Transformer blocks (same as GPT)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # Reward head: maps final hidden state to scalar reward
        # This replaces the language model head in GPT
        self.reward_head = nn.Linear(d_model, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for input sequences.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            rewards: Scalar rewards of shape (batch_size,)
        """
        # Get embeddings
        x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)

        # Pass through transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        # Final layer norm
        x = self.ln_f(x)  # (batch_size, seq_len, d_model)

        # Use the last token's hidden state for computing reward
        # This is standard practice - the last token summarizes the sequence
        last_hidden = x[:, -1, :]  # (batch_size, d_model)

        # Compute scalar reward
        reward = self.reward_head(last_hidden).squeeze(-1)  # (batch_size,)

        return reward

    def compute_preference_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the preference ranking loss.

        The reward model is trained to give higher rewards to "chosen" responses
        (preferred by humans) compared to "rejected" responses.

        Loss = -log(sigmoid(r_chosen - r_rejected))

        This is the Bradley-Terry model for pairwise comparisons.
        """
        r_chosen = self.forward(chosen_ids)
        r_rejected = self.forward(rejected_ids)

        # Pairwise ranking loss
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        return loss


# =============================================================================
# Part 2: Preference Dataset
# =============================================================================

class PreferenceDataset(Dataset):
    """
    Dataset for training the reward model.

    Each sample contains:
    - prompt: The input prompt
    - chosen: The preferred response (ranked higher by humans)
    - rejected: The less preferred response (ranked lower by humans)
    """

    def __init__(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str],
        tokenizer: SimpleTokenizer,
        max_length: int = 128,
    ):
        self.prompts = prompts
        self.chosen = chosen_responses
        self.rejected = rejected_responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        chosen = self.chosen[idx]
        rejected = self.rejected[idx]

        # Combine prompt + response
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected

        # Tokenize and pad/truncate
        chosen_ids = self._tokenize(chosen_text)
        rejected_ids = self._tokenize(rejected_text)

        return {
            'chosen_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_ids, dtype=torch.long),
        }

    def _tokenize(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        return tokens


# =============================================================================
# Part 3: PPO Trainer for RLHF
# =============================================================================

@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # PPO hyperparameters
    clip_epsilon: float = 0.2  # PPO clipping parameter
    vf_coef: float = 0.1  # Value function loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient

    # Training hyperparameters
    lr: float = 1e-5
    batch_size: int = 8
    mini_batch_size: int = 4
    ppo_epochs: int = 4  # Number of PPO optimization epochs per batch

    # KL penalty (to prevent policy from deviating too much from reference)
    kl_coef: float = 0.1
    target_kl: float = 0.01  # Target KL divergence

    # Generation parameters
    max_new_tokens: int = 32
    temperature: float = 1.0


class PPOTrainer:
    """
    PPO Trainer for RLHF.

    This implements Proximal Policy Optimization for fine-tuning language models
    using rewards from a reward model.

    Key components:
    1. Policy model (the LM we're training)
    2. Reference model (frozen copy of initial policy, for KL penalty)
    3. Value model (estimates expected future rewards)
    4. Reward model (scores generated responses)
    """

    def __init__(
        self,
        policy_model: SmallGPT,
        reward_model: RewardModel,
        tokenizer: SimpleTokenizer,
        config: PPOConfig,
        device: str = 'cpu',
    ):
        self.device = device
        self.config = config
        self.tokenizer = tokenizer

        # Policy model (the one we're training)
        self.policy = policy_model.to(device)

        # Reference model (frozen copy for KL penalty)
        self.ref_policy = copy.deepcopy(policy_model).to(device)
        self.ref_policy.eval()
        for param in self.ref_policy.parameters():
            param.requires_grad = False

        # Value head: simple linear layer that takes a scalar input and outputs value estimate
        # In a full implementation, this would be a separate network or share the policy backbone
        self.value_head = nn.Linear(1, 1).to(device)

        # Reward model (frozen, used to score responses)
        self.reward_model = reward_model.to(device)
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False

        # Optimizer (only policy parameters - value head is simplified)
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=config.lr,
        )

        # Statistics tracking
        self.stats = {
            'rewards': [],
            'kl_divergence': [],
            'policy_loss': [],
            'value_loss': [],
        }

    def generate_responses(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate responses for a batch of prompts.

        Returns:
            input_ids: Full sequences (prompt + response)
            response_masks: Binary mask indicating response tokens
            log_probs: Log probabilities of generated tokens
        """
        self.policy.eval()

        all_input_ids = []
        all_log_probs = []
        all_response_masks = []

        with torch.no_grad():
            for prompt in prompts:
                # Encode prompt
                prompt_ids = self.tokenizer.encode(prompt)
                prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(self.device)
                prompt_len = len(prompt_ids)

                # Generate response autoregressively
                generated_ids = prompt_tensor.clone()
                log_probs = []

                for _ in range(self.config.max_new_tokens):
                    logits = self.policy(generated_ids)
                    next_token_logits = logits[:, -1, :] / self.config.temperature

                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Compute log probability
                    log_prob = F.log_softmax(next_token_logits, dim=-1)
                    token_log_prob = log_prob.gather(1, next_token).squeeze(-1)
                    log_probs.append(token_log_prob)

                    # Append to sequence
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Create response mask (1 for response tokens, 0 for prompt)
                response_mask = torch.zeros(generated_ids.size(1), dtype=torch.bool)
                response_mask[prompt_len:] = True

                all_input_ids.append(generated_ids.squeeze(0))
                all_log_probs.append(torch.stack(log_probs).squeeze(-1))
                all_response_masks.append(response_mask)

        # Pad sequences to same length
        max_len = max(ids.size(0) for ids in all_input_ids)
        padded_ids = torch.zeros(len(prompts), max_len, dtype=torch.long, device=self.device)
        padded_masks = torch.zeros(len(prompts), max_len, dtype=torch.bool, device=self.device)

        for i, (ids, mask) in enumerate(zip(all_input_ids, all_response_masks)):
            padded_ids[i, :ids.size(0)] = ids
            padded_masks[i, :mask.size(0)] = mask

        # Stack log probs (already same length due to max_new_tokens)
        log_probs_tensor = torch.stack(all_log_probs)

        self.policy.train()
        return padded_ids, padded_masks, log_probs_tensor

    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        response_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rewards for generated sequences.

        This combines:
        1. Reward from the reward model
        2. KL penalty (to stay close to reference policy)
        """
        # Get reward from reward model
        with torch.no_grad():
            rewards = self.reward_model(input_ids)

        return rewards

    def compute_kl_penalty(
        self,
        input_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference policy.

        This prevents the policy from deviating too far from the original model,
        which helps maintain language quality and prevents reward hacking.
        """
        with torch.no_grad():
            # Get logits from both policies
            policy_logits = self.policy(input_ids)
            ref_logits = self.ref_policy(input_ids)

            # Compute KL divergence at each position
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # KL(policy || ref) = sum(policy * (log(policy) - log(ref)))
            kl = (torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)).sum(-1)

        return kl.mean(dim=-1)  # Average over sequence

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        For simplicity, we use single-step rewards here.
        In practice, you'd use GAE with a discount factor.
        """
        # Simple advantage: reward - value baseline
        advantages = rewards - values.squeeze(-1)

        # Normalize advantages (helps training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Returns for value function training
        returns = rewards

        return advantages, returns

    def ppo_step(
        self,
        input_ids: torch.Tensor,
        response_masks: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one PPO optimization step.

        This implements the clipped surrogate objective:
        L = min(r * A, clip(r, 1-eps, 1+eps) * A)

        where r = exp(log_prob - old_log_prob) is the probability ratio.
        """
        # Get current log probabilities
        logits = self.policy(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)

        # Extract log probs for the generated tokens
        # (This is simplified - in practice you'd gather specific token log probs)
        current_log_probs = log_probs[:, :-1, :].mean(dim=-1).mean(dim=-1)

        # Compute probability ratio
        # Ensure shapes match
        if old_log_probs.dim() > 1:
            old_log_probs_scalar = old_log_probs.mean(dim=-1)
        else:
            old_log_probs_scalar = old_log_probs

        ratio = torch.exp(current_log_probs - old_log_probs_scalar)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss (simplified - just MSE between predicted and actual returns)
        # In a full implementation, value predictions would come from a value network
        value_loss = F.mse_loss(current_log_probs, returns)

        # Entropy bonus (encourages exploration)
        # Compute entropy properly: -sum(p * log(p))
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(-1).mean()

        # Total loss
        loss = (
            policy_loss
            + self.config.vf_coef * value_loss
            - self.config.entropy_coef * entropy
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item(),
        }

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        Perform one complete PPO training step.

        1. Generate responses for prompts
        2. Compute rewards
        3. Compute advantages
        4. Optimize policy with PPO
        """
        # Step 1: Generate responses
        input_ids, response_masks, old_log_probs = self.generate_responses(prompts)

        # Step 2: Compute rewards
        rewards = self.compute_rewards(input_ids, response_masks)
        kl_penalty = self.compute_kl_penalty(input_ids, old_log_probs)

        # Apply KL penalty to rewards
        adjusted_rewards = rewards - self.config.kl_coef * kl_penalty

        # Step 3: Compute value estimates and advantages
        with torch.no_grad():
            # Simplified value estimate
            values = torch.zeros_like(rewards)

        advantages, returns = self.compute_advantages(adjusted_rewards, values)

        # Step 4: PPO optimization (multiple epochs)
        total_stats = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'total_loss': 0}

        for _ in range(self.config.ppo_epochs):
            stats = self.ppo_step(input_ids, response_masks, old_log_probs, advantages, returns)
            for k, v in stats.items():
                total_stats[k] += v / self.config.ppo_epochs

        # Track statistics
        self.stats['rewards'].append(rewards.mean().item())
        self.stats['kl_divergence'].append(kl_penalty.mean().item())
        self.stats['policy_loss'].append(total_stats['policy_loss'])
        self.stats['value_loss'].append(total_stats['value_loss'])

        return {
            'mean_reward': rewards.mean().item(),
            'mean_kl': kl_penalty.mean().item(),
            **total_stats,
        }


# =============================================================================
# Part 4: Reward Model Training
# =============================================================================

def train_reward_model(
    reward_model: RewardModel,
    train_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = 'cpu',
) -> List[float]:
    """
    Train the reward model on human preference data.

    The reward model learns to predict which response humans would prefer.
    """
    reward_model = reward_model.to(device)
    optimizer = AdamW(reward_model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Reward Model Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            chosen_ids = batch['chosen_ids'].to(device)
            rejected_ids = batch['rejected_ids'].to(device)

            optimizer.zero_grad()
            loss = reward_model.compute_preference_loss(chosen_ids, rejected_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")

    return losses


# =============================================================================
# Part 5: Example Usage
# =============================================================================

def create_sample_preference_data(tokenizer: SimpleTokenizer):
    """
    Create synthetic preference data for demonstration.

    In practice, this data would come from human annotators who compare
    pairs of model responses and indicate which one is better.
    """
    # Example prompts and preferences
    # Format: (prompt, preferred_response, less_preferred_response)
    examples = [
        (
            "What is the weather like? ",
            "The weather today is sunny and pleasant with clear skies.",
            "weather good maybe rain idk",
        ),
        (
            "Tell me about cats. ",
            "Cats are wonderful domestic animals known for their independence and grace.",
            "cats meow lol",
        ),
        (
            "How do I make tea? ",
            "To make tea, boil water, steep the tea bag for three to five minutes, then enjoy.",
            "put stuff in water",
        ),
        (
            "What is programming? ",
            "Programming is the process of creating instructions for computers to execute tasks.",
            "typing on keyboard",
        ),
        (
            "Describe the ocean. ",
            "The ocean is a vast body of saltwater covering most of Earth's surface.",
            "water big blue wet",
        ),
    ]

    # Duplicate and shuffle for more training data
    examples = examples * 10

    prompts = [ex[0] for ex in examples]
    chosen = [ex[1] for ex in examples]
    rejected = [ex[2] for ex in examples]

    return prompts, chosen, rejected


def main():
    """
    Main function demonstrating the complete RLHF pipeline.
    """
    print("=" * 70)
    print("Reinforcement Learning for LLM Training Tutorial")
    print("=" * 70)

    # Set random seed
    set_seed(42)
    device = get_device()
    print(f"\nUsing device: {device}")

    # Configuration
    model_config = {
        'vocab_size': 256,  # Character-level for simplicity
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'max_seq_len': 128,
        'dropout': 0.1,
    }

    # Create tokenizer (character-level for demonstration)
    tokenizer = SimpleTokenizer()
    # Build vocab from a sample text
    sample_text = """
    The weather today is sunny and pleasant with clear skies.
    Cats are wonderful domestic animals known for their independence.
    Programming is the process of creating instructions for computers.
    The ocean is a vast body of saltwater covering most of Earth.
    """
    tokenizer.build_vocab(sample_text)
    model_config['vocab_size'] = tokenizer.vocab_size

    print(f"\nVocabulary size: {tokenizer.vocab_size}")

    # =========================================================================
    # Step 1: Create models
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Creating Models")
    print("=" * 70)

    # Policy model (the LM we'll fine-tune)
    policy_model = create_small_gpt(model_config)
    print(f"Policy model parameters: {sum(p.numel() for p in policy_model.parameters()):,}")

    # Reward model (learns to score responses)
    reward_model = RewardModel(**model_config)
    print(f"Reward model parameters: {sum(p.numel() for p in reward_model.parameters()):,}")

    # =========================================================================
    # Step 2: Train the Reward Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Training the Reward Model")
    print("=" * 70)
    print("\nThe reward model learns to predict human preferences.")
    print("It's trained on pairs of responses where one is preferred over the other.\n")

    # Create preference dataset
    prompts, chosen, rejected = create_sample_preference_data(tokenizer)
    pref_dataset = PreferenceDataset(
        prompts, chosen, rejected, tokenizer, max_length=64
    )
    pref_loader = DataLoader(pref_dataset, batch_size=8, shuffle=True)

    # Train reward model
    print("Training reward model on preference data...")
    rm_losses = train_reward_model(
        reward_model, pref_loader, epochs=3, lr=1e-3, device=device
    )

    # =========================================================================
    # Step 3: RLHF with PPO
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Fine-tuning with PPO (RLHF)")
    print("=" * 70)
    print("\nNow we use PPO to optimize the policy model to generate responses")
    print("that the reward model scores highly.\n")

    # Create PPO trainer
    ppo_config = PPOConfig(
        lr=1e-4,
        batch_size=4,
        mini_batch_size=4,
        ppo_epochs=2,
        max_new_tokens=16,
        kl_coef=0.1,
    )

    ppo_trainer = PPOTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=ppo_config,
        device=device,
    )

    # Training prompts
    training_prompts = [
        "What is ",
        "Tell me ",
        "How do ",
        "Describe ",
    ]

    # PPO training loop
    num_steps = 10
    print(f"Running {num_steps} PPO training steps...\n")

    for step in range(num_steps):
        stats = ppo_trainer.train_step(training_prompts)
        print(f"Step {step + 1:3d} | "
              f"Reward: {stats['mean_reward']:7.4f} | "
              f"KL: {stats['mean_kl']:7.4f} | "
              f"Policy Loss: {stats['policy_loss']:7.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    print("""
    Summary of what we did:

    1. REWARD MODEL TRAINING:
       - Created a reward model that shares architecture with the policy
       - Trained it on preference pairs (chosen vs rejected responses)
       - The model learned to give higher scores to preferred responses

    2. PPO FINE-TUNING:
       - Used PPO to optimize the policy to maximize rewards
       - Added KL penalty to prevent the policy from deviating too much
       - This balances reward maximization with staying close to the original model

    Key RLHF Concepts Demonstrated:

    - Reward Modeling: Learning human preferences from comparison data
    - PPO Clipping: Prevents too-large policy updates for stability
    - KL Penalty: Keeps the model from "reward hacking"
    - Value Function: Provides baseline for variance reduction
    - Advantage Estimation: Measures how much better an action is than expected

    This is a simplified version of what's used to train models like ChatGPT!
    """)

    # Show final statistics
    print("\nTraining Statistics:")
    print(f"  Final mean reward: {ppo_trainer.stats['rewards'][-1]:.4f}")
    print(f"  Final KL divergence: {ppo_trainer.stats['kl_divergence'][-1]:.4f}")
    print(f"  Reward model final loss: {rm_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
