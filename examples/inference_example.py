#!/usr/bin/env python3
"""
Example script for using a trained GPT model for text generation.
This script demonstrates how to load a model and generate text.
"""

import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gpt import create_small_gpt
from training.dataset import SimpleTokenizer
from utils.inference import GPTInference
from utils.helpers import get_device


def main():
    # Configuration (should match your trained model)
    config = {
        'vocab_size': 1000,  # Will be updated based on tokenizer
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 512,
        'max_seq_len': 256,
        'dropout': 0.1,
    }

    device = get_device()
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer_path = 'checkpoints/tokenizer.pkl'

    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Please run train_example.py first to create a trained model.")
        return

    tokenizer.load(tokenizer_path)
    config['vocab_size'] = tokenizer.vocab_size
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create and load model
    print("Loading model...")
    model = create_small_gpt(config)

    checkpoint_path = 'checkpoints/best_model.pt'
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Please run train_example.py first to create a trained model.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print("Model loaded successfully!")

    # Create inference object
    inference = GPTInference(model, tokenizer, device)

    # Example prompts
    prompts = [
        "Once upon a time",
        "The programmer",
        "In a land far away",
        "Through perseverance"
    ]

    print("\nGenerating text...")
    print("=" * 60)

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)

        # Generate with different parameters
        generations = [
            {
                "name": "Greedy (temp=0.1)",
                "params": {"temperature": 0.1, "max_new_tokens": 50}
            },
            {
                "name": "Balanced (temp=0.7)",
                "params": {"temperature": 0.7, "max_new_tokens": 50}
            },
            {
                "name": "Creative (temp=1.2)",
                "params": {"temperature": 1.2, "max_new_tokens": 50}
            },
            {
                "name": "Top-k sampling",
                "params": {"temperature": 0.8, "top_k": 10, "max_new_tokens": 50}
            }
        ]

        for gen_config in generations:
            generated = inference.generate(prompt, **gen_config["params"])
            print(f"{gen_config['name']}: {generated}")
            print()

    # Interactive generation
    print("\nInteractive mode (type 'quit' to exit):")
    print("=" * 60)

    while True:
        user_prompt = input("\nEnter a prompt: ").strip()
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            break

        if user_prompt:
            try:
                generated = inference.generate(
                    user_prompt,
                    max_new_tokens=100,
                    temperature=0.8,
                    top_k=20
                )
                print(f"Generated: {generated}")
            except Exception as e:
                print(f"Error generating text: {e}")

    print("Goodbye!")


if __name__ == "__main__":
    main()