#!/usr/bin/env python3
"""
LoRA Fine-Tuning Tutorial
=========================

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique
that dramatically reduces memory requirements and training time.

Key Concepts:
- Instead of updating all model weights, LoRA adds small trainable matrices
- Original weights stay frozen (no forgetting!)
- Typically trains only 0.1-1% of parameters
- Can fine-tune 7B+ models on consumer GPUs

How it works:
- For a weight matrix W (d x k), instead of learning W' directly:
- Learn two small matrices: A (d x r) and B (r x k) where r << d, k
- Output = W*x + (A*B)*x
- Only A and B are trained (much fewer parameters!)

Prerequisites:
    pip install transformers peft accelerate bitsandbytes datasets
"""

import os
import torch
from typing import Dict, Any, Optional


def check_dependencies():
    """Check if required packages are installed."""
    required = ['transformers', 'peft', 'datasets']
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
# Part 1: Understanding LoRA Configuration
# =============================================================================

def explain_lora_config():
    """
    Explain the key LoRA hyperparameters.
    """
    print("\n" + "=" * 60)
    print("LoRA Configuration Explained")
    print("=" * 60)

    config_explanation = """
    Key LoRA Parameters:
    --------------------

    r (rank): 4-64 typically
        - The rank of the low-rank matrices
        - Higher = more capacity but more parameters
        - Start with 8 or 16, increase if underfitting

    lora_alpha: Usually 16-32
        - Scaling factor for LoRA weights
        - Effective learning rate scales with alpha/r
        - Common: alpha = 2*r

    target_modules: Which layers to apply LoRA
        - For transformers: ["q_proj", "v_proj"] (attention)
        - More aggressive: ["q_proj", "k_proj", "v_proj", "o_proj"]
        - Even more: Include FFN layers ["gate_proj", "up_proj", "down_proj"]

    lora_dropout: 0.0-0.1
        - Dropout applied to LoRA layers
        - Helps prevent overfitting

    bias: "none", "all", or "lora_only"
        - Whether to train bias parameters
        - "none" is most common

    task_type: "CAUSAL_LM", "SEQ_CLS", etc.
        - Must match your task
        - CAUSAL_LM for text generation
    """
    print(config_explanation)


# =============================================================================
# Part 2: Basic LoRA Setup
# =============================================================================

def setup_lora_model():
    """
    Set up a model with LoRA adapters.
    """
    print("\n" + "=" * 60)
    print("Setting Up LoRA Model")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    # Use a small model for demonstration
    model_name = "distilgpt2"

    print(f"\nLoading base model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    print(f"Base model parameters: {model.num_parameters():,}")

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,                        # Rank
        lora_alpha=16,              # Scaling factor
        target_modules=["c_attn"],  # GPT-2 uses "c_attn" for attention
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print(f"\nLoRA Config:")
    print(f"  - Rank (r): {lora_config.r}")
    print(f"  - Alpha: {lora_config.lora_alpha}")
    print(f"  - Target modules: {lora_config.target_modules}")
    print(f"  - Dropout: {lora_config.lora_dropout}")

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nAfter LoRA:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Trainable %: {100 * trainable_params / total_params:.2f}%")

    return model, tokenizer, lora_config


# =============================================================================
# Part 3: Training with LoRA
# =============================================================================

def create_sample_dataset():
    """Create a sample dataset for fine-tuning."""
    from datasets import Dataset

    data = [
        {"text": "Question: What is Python?\nAnswer: Python is a high-level programming language known for its simplicity and readability."},
        {"text": "Question: Explain machine learning.\nAnswer: Machine learning is a branch of AI where computers learn patterns from data without explicit programming."},
        {"text": "Question: What is a database?\nAnswer: A database is an organized collection of structured data stored electronically and accessed via a database management system."},
        {"text": "Question: Define API.\nAnswer: An API (Application Programming Interface) is a set of protocols that allows different software applications to communicate."},
        {"text": "Question: What is cloud computing?\nAnswer: Cloud computing delivers computing services over the internet, including storage, processing, and software."},
    ] * 20  # 100 examples

    return Dataset.from_list(data)


def train_with_lora():
    """
    Complete LoRA training pipeline.
    """
    print("\n" + "=" * 60)
    print("Training with LoRA")
    print("=" * 60)

    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import get_peft_model, LoraConfig, TaskType

    # Setup model
    model, tokenizer, lora_config = setup_lora_model()

    # Prepare dataset
    print("\nPreparing dataset...")
    dataset = create_sample_dataset()

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    split = tokenized_dataset.train_test_split(test_size=0.1)

    print(f"Training examples: {len(split['train'])}")
    print(f"Evaluation examples: {len(split['test'])}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments (optimized for LoRA)
    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Can use larger batches with LoRA!
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,  # LoRA can use higher LR
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nReady to train!")
    print("(Uncomment trainer.train() to actually run training)")

    # Uncomment to train:
    # trainer.train()

    # Save LoRA adapters only (very small!)
    # model.save_pretrained("./lora_adapters")

    return model, tokenizer


# =============================================================================
# Part 4: QLoRA (Quantized LoRA) - For Large Models
# =============================================================================

def setup_qlora():
    """
    Set up QLoRA for training large models on consumer hardware.

    QLoRA combines:
    - 4-bit quantization (reduces memory ~4x)
    - LoRA (trains tiny fraction of parameters)

    This allows fine-tuning 7B-70B models on a single GPU!
    """
    print("\n" + "=" * 60)
    print("QLoRA Setup (for Large Models)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\nQLoRA requires CUDA. Showing configuration only...")
        show_qlora_config()
        return None, None

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    except ImportError:
        print("Install bitsandbytes: pip install bitsandbytes")
        return None, None

    # For demonstration, use a small model
    # In practice, use larger models like:
    # - "meta-llama/Llama-2-7b-hf"
    # - "mistralai/Mistral-7B-v0.1"
    # - "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"\nLoading {model_name} with 4-bit quantization...")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # Normalized float 4-bit
        bnb_4bit_compute_dtype=torch.float16, # Compute in fp16
        bnb_4bit_use_double_quant=True,      # Nested quantization
    )

    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config for QLoRA
    lora_config = LoraConfig(
        r=16,                   # Slightly higher rank for large models
        lora_alpha=32,
        target_modules=[        # Target attention and FFN
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print stats
    print(f"\nModel memory: {model.get_memory_footprint() / 1024**2:.1f} MB")
    model.print_trainable_parameters()

    return model, tokenizer


def show_qlora_config():
    """Show QLoRA configuration without running it."""
    config_code = '''
    # QLoRA Configuration Example

    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Memory comparison:
    # - Llama-2-7B full precision: ~28 GB
    # - Llama-2-7B 4-bit: ~4 GB
    # - Llama-2-7B 4-bit + LoRA training: ~6-8 GB
    '''
    print(config_code)


# =============================================================================
# Part 5: Loading and Using LoRA Adapters
# =============================================================================

def using_lora_adapters():
    """
    How to save, load, and use LoRA adapters.
    """
    print("\n" + "=" * 60)
    print("Using LoRA Adapters")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType

    model_name = "distilgpt2"

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a LoRA model (simulating a trained one)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        task_type=TaskType.CAUSAL_LM,
    )
    lora_model = get_peft_model(base_model, lora_config)

    print("\n1. Saving LoRA adapters (NOT the full model):")
    print("   model.save_pretrained('./my_lora_adapters')")
    print("   # This creates a small folder (~few MB)")

    # In practice:
    # lora_model.save_pretrained("./my_lora_adapters")

    print("\n2. Loading LoRA adapters onto base model:")
    print("   base_model = AutoModelForCausalLM.from_pretrained('gpt2')")
    print("   model = PeftModel.from_pretrained(base_model, './my_lora_adapters')")

    # In practice:
    # loaded_model = PeftModel.from_pretrained(base_model, "./my_lora_adapters")

    print("\n3. Merging adapters into base model (for deployment):")
    print("   merged_model = model.merge_and_unload()")
    print("   merged_model.save_pretrained('./merged_model')")
    print("   # Now it's a regular model, no PEFT dependency needed")

    print("\n4. Switching between adapters:")
    print("   model.load_adapter('./adapter_1', adapter_name='task1')")
    print("   model.load_adapter('./adapter_2', adapter_name='task2')")
    print("   model.set_adapter('task1')  # Use task1 adapter")
    print("   model.set_adapter('task2')  # Switch to task2")

    # Quick inference test
    print("\n" + "-" * 40)
    print("Quick inference test:")

    prompt = "Question: What is AI?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = lora_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    print(f"\nPrompt: {prompt}")
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the LoRA tutorial."""
    print("=" * 60)
    print("LoRA Fine-Tuning Tutorial")
    print("=" * 60)

    if not check_dependencies():
        return

    # Run tutorial sections
    explain_lora_config()
    model, tokenizer = train_with_lora()
    setup_qlora()
    using_lora_adapters()

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print("""
    Summary:
    --------
    1. LoRA trains only 0.1-1% of parameters
    2. Much faster and uses less memory
    3. QLoRA adds 4-bit quantization for large models
    4. Adapters are small and easy to share
    5. Can merge adapters back into base model

    Next steps:
    - Try training with your own data
    - Experiment with different ranks (r) and alpha values
    - Use QLoRA for 7B+ models
    - Check out the TRL library for RLHF
    """)


if __name__ == "__main__":
    main()
