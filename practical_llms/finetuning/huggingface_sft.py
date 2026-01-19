#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) with HuggingFace
=============================================

This tutorial demonstrates how to fine-tune a pre-trained language model
on your own data using HuggingFace's Trainer API.

What is SFT?
- Start with a pre-trained model (knows language)
- Fine-tune on your specific task/domain
- Model learns to follow your data's patterns

Use cases:
- Custom chatbots
- Domain-specific assistants (legal, medical, etc.)
- Instruction following
- Task-specific models

Prerequisites:
    pip install transformers datasets accelerate
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


def check_dependencies():
    """Check if required packages are installed."""
    required = ['transformers', 'datasets', 'accelerate']
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
# Part 1: Preparing Your Dataset
# =============================================================================

def create_sample_dataset():
    """
    Create a sample instruction-following dataset.

    In practice, you would load your own data from:
    - JSON/JSONL files
    - CSV files
    - HuggingFace datasets hub
    - Databases
    """
    from datasets import Dataset

    # Sample instruction-following data
    # Format: instruction -> response pairs
    data = [
        {
            "instruction": "Explain what a neural network is in simple terms.",
            "response": "A neural network is like a simplified model of the human brain. It consists of layers of connected nodes (neurons) that process information. Data flows through these layers, with each connection having a weight that determines its importance. The network learns by adjusting these weights based on examples, gradually improving its ability to recognize patterns and make predictions."
        },
        {
            "instruction": "Write a Python function to reverse a string.",
            "response": """Here's a Python function to reverse a string:

```python
def reverse_string(s):
    return s[::-1]
```

This uses Python's slice notation with a step of -1 to reverse the string. You can also use:

```python
def reverse_string(s):
    return ''.join(reversed(s))
```"""
        },
        {
            "instruction": "What are the benefits of exercise?",
            "response": "Regular exercise offers numerous benefits: 1) Improves cardiovascular health and reduces heart disease risk, 2) Helps maintain a healthy weight, 3) Strengthens muscles and bones, 4) Boosts mental health by reducing anxiety and depression, 5) Increases energy levels, 6) Improves sleep quality, 7) Enhances cognitive function and memory, 8) Strengthens the immune system."
        },
        {
            "instruction": "Summarize the water cycle.",
            "response": "The water cycle is the continuous movement of water on Earth. It begins with evaporation, where the sun heats water in oceans and lakes, turning it into vapor. This vapor rises and cools, forming clouds through condensation. When clouds become saturated, precipitation occurs as rain or snow. Water then flows through rivers back to oceans or seeps into groundwater, and the cycle repeats."
        },
        {
            "instruction": "How do I make a good cup of coffee?",
            "response": "To make a great cup of coffee: 1) Start with fresh, quality beans, 2) Grind just before brewing for best flavor, 3) Use the right ratio - about 2 tablespoons per 6 ounces of water, 4) Use water between 195-205°F (not boiling), 5) Brew for 4-5 minutes for drip methods, 6) Use filtered water for cleaner taste, 7) Clean your equipment regularly, 8) Serve immediately for optimal flavor."
        },
    ]

    # Duplicate for more training data (in practice, you'd have more examples)
    data = data * 20  # 100 examples

    return Dataset.from_list(data)


def format_instruction(example: Dict[str, str]) -> str:
    """
    Format an instruction-response pair into the training format.

    This creates the text that the model will learn to generate.
    """
    return f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""


def prepare_dataset_for_training(dataset, tokenizer, max_length: int = 512):
    """
    Tokenize and prepare the dataset for training.
    """

    def tokenize_function(examples):
        # Format each example
        texts = [format_instruction({"instruction": inst, "response": resp})
                for inst, resp in zip(examples["instruction"], examples["response"])]

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,  # Return lists, not tensors
        )

        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset


# =============================================================================
# Part 2: Setting Up the Trainer
# =============================================================================

def setup_training():
    """
    Set up model, tokenizer, and training configuration.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )

    print("=" * 60)
    print("Supervised Fine-Tuning Tutorial")
    print("=" * 60)

    # Model selection
    # Small models for learning/testing:
    #   - "gpt2" (~500MB)
    #   - "distilgpt2" (~350MB)
    # Medium models (need GPU):
    #   - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (~2GB)
    #   - "microsoft/phi-2" (~5GB)
    # Large models (need good GPU):
    #   - "meta-llama/Llama-2-7b-hf" (~14GB)
    #   - "mistralai/Mistral-7B-v0.1" (~14GB)

    model_name = "distilgpt2"  # Small model for demonstration

    print(f"\nLoading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token (required for training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float16 on GPU for efficiency
    )

    # Resize embeddings if we added tokens
    model.resize_token_embeddings(len(tokenizer))

    print(f"Model loaded! Parameters: {model.num_parameters():,}")

    # Create dataset
    print("\nPreparing dataset...")
    dataset = create_sample_dataset()
    tokenized_dataset = prepare_dataset_for_training(dataset, tokenizer)

    # Split into train/eval
    split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")

    # Data collator handles batching and dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )

    # Training arguments
    output_dir = "./sft_output"
    training_args = TrainingArguments(
        output_dir=output_dir,

        # Training hyperparameters
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16

        # Learning rate schedule
        learning_rate=5e-5,
        warmup_steps=100,
        lr_scheduler_type="cosine",

        # Optimization
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,  # Keep only 2 checkpoints

        # Efficiency
        fp16=torch.cuda.is_available(),  # Mixed precision on GPU
        dataloader_num_workers=0,

        # Misc
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb/tensorboard for simplicity
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer, tokenizer, model


# =============================================================================
# Part 3: Training
# =============================================================================

def train_model(trainer):
    """
    Run the training loop.
    """
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    # Train!
    train_result = trainer.train()

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Training time: {train_result.metrics['train_runtime']:.1f}s")
    print(f"Samples/second: {train_result.metrics['train_samples_per_second']:.1f}")

    # Save the final model
    trainer.save_model("./sft_final_model")
    print("\nModel saved to ./sft_final_model")

    return train_result


# =============================================================================
# Part 4: Inference with Fine-tuned Model
# =============================================================================

def test_finetuned_model(model, tokenizer):
    """
    Test the fine-tuned model with new instructions.
    """
    print("\n" + "=" * 60)
    print("Testing Fine-tuned Model")
    print("=" * 60)

    model.eval()
    device = next(model.parameters()).device

    test_instructions = [
        "Explain what recursion is in programming.",
        "What are the main differences between Python and JavaScript?",
        "How do I stay motivated while learning to code?",
    ]

    for instruction in test_instructions:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\n{'='*40}")
        print(f"Instruction: {instruction}")
        print(f"\nGenerated Response:")
        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        print(response[:500])  # Truncate for display


# =============================================================================
# Part 5: Complete Training Pipeline
# =============================================================================

def main():
    """Run the complete fine-tuning pipeline."""

    if not check_dependencies():
        return

    # Setup
    trainer, tokenizer, model = setup_training()

    # Check if we should actually train (it takes time)
    print("\n" + "=" * 60)
    print("Ready to train!")
    print("=" * 60)
    print("""
    This will fine-tune a small GPT-2 model on instruction data.

    Expected time:
    - CPU: ~5-10 minutes
    - GPU: ~1-2 minutes

    The model will learn to follow instructions and generate responses
    in a similar style to the training data.
    """)

    # Uncomment to actually train:
    # train_result = train_model(trainer)

    # For demonstration, just show what would happen
    print("To start training, uncomment the train_model() call in main()")
    print("\nShowing a quick evaluation of the base model:")

    test_finetuned_model(model, tokenizer)

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print("""
    Next steps:
    1. Uncomment train_model() to actually fine-tune
    2. Collect more training data for your use case
    3. Try larger models for better quality
    4. Use LoRA (see lora_finetuning.py) for efficient fine-tuning
    5. Use TRL library (see rlhf_trl.py) for RLHF
    """)


if __name__ == "__main__":
    main()
