#!/usr/bin/env python3
"""
Example script for training a small GPT model from scratch.
This script demonstrates how to prepare data, create a model, and train it.
"""

import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gpt import create_small_gpt
from training.dataset import prepare_data, create_dataloaders
from training.trainer import GPTTrainer
from utils.helpers import set_seed, print_model_info, get_device


def create_sample_data():
    """Create a sample text file for training if it doesn't exist."""
    sample_text = """
    Once upon a time, in a land far away, there lived a curious programmer who wanted to understand how language models work.
    This programmer decided to build a small GPT model from scratch to learn about transformers, attention mechanisms, and neural networks.
    The journey was challenging but rewarding, as each line of code brought new insights into the fascinating world of artificial intelligence.
    Through perseverance and dedication, the programmer discovered the beauty of self-attention and the power of neural language generation.
    And so, the adventure in machine learning continued, with each experiment revealing new mysteries to explore.
    """ * 20  # Repeat to have more training data

    os.makedirs('data', exist_ok=True)
    with open('data/sample_text.txt', 'w') as f:
        f.write(sample_text)

    return 'data/sample_text.txt'


def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Configuration
    config = {
        'vocab_size': 1000,  # Will be updated based on actual vocabulary
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 512,
        'max_seq_len': 256,
        'dropout': 0.1,
    }

    # Training configuration
    batch_size = 16
    learning_rate = 1e-3
    epochs = 5
    block_size = 64

    print("Creating sample data...")
    text_file = create_sample_data()

    print("Preparing datasets...")
    train_dataset, val_dataset, tokenizer = prepare_data(
        text_file, block_size=block_size, train_split=0.9
    )

    # Update vocab size based on actual tokenizer
    config['vocab_size'] = tokenizer.vocab_size
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, batch_size=batch_size
    )

    print("Creating model...")
    model = create_small_gpt(config)
    print_model_info(model)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Create trainer
    trainer = GPTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=learning_rate,
        device=device
    )

    print("\nStarting training...")
    train_losses, val_losses = trainer.train(
        epochs=epochs,
        save_dir='checkpoints',
        save_every=2
    )

    print("\nTraining completed!")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")

    # Save tokenizer
    os.makedirs('checkpoints', exist_ok=True)
    tokenizer.save('checkpoints/tokenizer.pkl')
    print("Tokenizer saved to checkpoints/tokenizer.pkl")


if __name__ == "__main__":
    main()