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
    pip install transformers peft accelerate datasets

Usage:
    python lora_finetuning.py           # Show tutorial (no training)
    python lora_finetuning.py --train   # Actually run training (1 epoch)
"""

import os
import sys
import argparse
import torch
from typing import Dict, Any, Optional

# Global flag for whether to actually run training
RUN_TRAINING = False


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


def detect_gpu_vram():
    """Detect available GPU VRAM and return optimization recommendations."""
    if not torch.cuda.is_available():
        return {
            'device': 'cpu',
            'vram_gb': 0,
            'batch_size': 2,
            'use_fp16': False,
            'use_8bit': False,
            'gradient_checkpointing': True,
            'max_length': 256,
        }

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f"\n[GPU Detected] {torch.cuda.get_device_name(0)}")
    print(f"[VRAM] {vram_gb:.1f} GB")

    # Optimize settings based on VRAM
    if vram_gb < 4:  # Very small GPU
        config = {
            'device': 'cuda',
            'vram_gb': vram_gb,
            'batch_size': 1,
            'use_fp16': True,
            'use_8bit': True,
            'gradient_checkpointing': True,
            'max_length': 128,
            'recommendation': 'Using 8-bit + small batches for <4GB VRAM'
        }
    elif vram_gb < 8:  # Small GPU (GTX 1060, RTX 3050)
        config = {
            'device': 'cuda',
            'vram_gb': vram_gb,
            'batch_size': 2,
            'use_fp16': True,
            'use_8bit': False,
            'gradient_checkpointing': True,
            'max_length': 256,
            'recommendation': 'Using FP16 + gradient checkpointing for 4-8GB VRAM'
        }
    elif vram_gb < 12:  # Medium GPU (RTX 3060, RTX 2080)
        config = {
            'device': 'cuda',
            'vram_gb': vram_gb,
            'batch_size': 4,
            'use_fp16': True,
            'use_8bit': False,
            'gradient_checkpointing': False,
            'max_length': 512,
            'recommendation': 'Standard settings for 8-12GB VRAM'
        }
    else:  # Large GPU
        config = {
            'device': 'cuda',
            'vram_gb': vram_gb,
            'batch_size': 8,
            'use_fp16': True,
            'use_8bit': False,
            'gradient_checkpointing': False,
            'max_length': 512,
            'recommendation': 'High performance settings for 12GB+ VRAM'
        }

    print(f"[Optimization] {config['recommendation']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - FP16: {config['use_fp16']}")
    print(f"  - 8-bit: {config['use_8bit']}")
    print(f"  - Gradient checkpointing: {config['gradient_checkpointing']}")

    return config


# =============================================================================
# Part 1: Understanding LoRA Configuration
# =============================================================================

def explain_lora_config():
    """
    Explain the key LoRA hyperparameters.
    """
    print("\n" + "=" * 60)
    print("Part 1: LoRA Configuration Explained")
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

    Memory Savings:
    ---------------
    LoRA typically trains only 0.1-1% of model parameters!

    Example for 7B model:
    - Full fine-tuning: 7B parameters
    - LoRA (r=8): ~8M parameters (0.1%)
    - LoRA (r=64): ~67M parameters (1%)
    """
    print(config_explanation)


# =============================================================================
# Part 2: Dataset Options
# =============================================================================

def load_dataset_option():
    """Load dataset - either HuggingFace or synthetic."""
    from datasets import load_dataset, Dataset

    print("\n" + "=" * 60)
    print("Part 2: Loading Dataset")
    print("=" * 60)

    print("\nDataset options:")
    print("  1. Synthetic Q&A (fast, no download)")
    print("  2. HuggingFace: databricks/databricks-dolly-15k (instruction following)")
    print("  3. HuggingFace: OpenAssistant conversations")

    # Use synthetic for demonstration
    print("\nUsing: Synthetic Q&A dataset (for speed)")

    data = [
        {"text": "Question: What is Python?\nAnswer: Python is a high-level programming language known for its simplicity and readability."},
        {"text": "Question: Explain machine learning.\nAnswer: Machine learning is a branch of AI where computers learn patterns from data without explicit programming."},
        {"text": "Question: What is a database?\nAnswer: A database is an organized collection of structured data stored electronically and accessed via a database management system."},
        {"text": "Question: Define API.\nAnswer: An API (Application Programming Interface) is a set of protocols that allows different software applications to communicate."},
        {"text": "Question: What is cloud computing?\nAnswer: Cloud computing delivers computing services over the internet, including storage, processing, and software."},
        {"text": "Question: Explain Docker.\nAnswer: Docker is a platform for developing, shipping, and running applications in isolated containers."},
        {"text": "Question: What is Git?\nAnswer: Git is a distributed version control system for tracking changes in source code during software development."},
        {"text": "Question: Define REST API.\nAnswer: REST API is an architectural style for building web services that use HTTP requests to access and manipulate data."},
    ] * 15  # 120 examples

    dataset = Dataset.from_list(data)

    print(f"\nDataset loaded: {len(dataset)} examples")
    print(f"Sample: {data[0]['text'][:100]}...")

    # Show how to load HuggingFace datasets
    print("\n" + "-" * 40)
    print("To use HuggingFace datasets:")
    print("  dataset = load_dataset('databricks/databricks-dolly-15k', split='train')")
    print("  # Format the data appropriately for your task")

    return dataset


# =============================================================================
# Part 3: Basic LoRA Setup
# =============================================================================

def setup_lora_model(gpu_config):
    """
    Set up a model with LoRA adapters.
    """
    print("\n" + "=" * 60)
    print("Part 3: Setting Up LoRA Model")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    # Use a small model for demonstration
    model_name = "distilgpt2"  # ~82M params, works on CPU/small GPU

    print(f"\nLoading base model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with appropriate settings
    load_kwargs = {
        'pretrained_model_name_or_path': model_name,
        'torch_dtype': torch.float16 if gpu_config['use_fp16'] else torch.float32,
    }

    # Add 8-bit quantization if needed (for very small VRAM)
    if gpu_config['use_8bit']:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs['device_map'] = 'auto'
            print("  Using 8-bit quantization for memory efficiency")
        except ImportError:
            print("  Install bitsandbytes for 8-bit: pip install bitsandbytes")
            gpu_config['use_8bit'] = False

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    # Prepare model for k-bit training if quantized
    if gpu_config['use_8bit']:
        model = prepare_model_for_kbit_training(model)

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

    # Enable gradient checkpointing for memory efficiency
    if gpu_config['gradient_checkpointing']:
        model.enable_input_require_grads()
        if hasattr(model.base_model, 'gradient_checkpointing_enable'):
            model.base_model.gradient_checkpointing_enable()
            print("  - Gradient checkpointing: Enabled")

    return model, tokenizer, lora_config


# =============================================================================
# Part 4: Training with LoRA
# =============================================================================

def train_with_lora():
    """
    Complete LoRA training pipeline.
    """
    print("\n" + "=" * 60)
    print("Part 4: Training with LoRA")
    print("=" * 60)

    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    # Detect GPU and get optimized config
    gpu_config = detect_gpu_vram()

    # Setup model
    model, tokenizer, lora_config = setup_lora_model(gpu_config)

    # Prepare dataset
    print("\nPreparing dataset...")
    dataset = load_dataset_option()

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=gpu_config['max_length'],
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

    # Training arguments (optimized for LoRA and GPU)
    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=1,  # Quick training for demo
        per_device_train_batch_size=gpu_config['batch_size'],
        per_device_eval_batch_size=gpu_config['batch_size'],
        gradient_accumulation_steps=4 // gpu_config['batch_size'],  # Effective batch size of 4
        learning_rate=2e-4,  # LoRA can use higher LR
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        fp16=gpu_config['use_fp16'],
        report_to="none",
        gradient_checkpointing=gpu_config['gradient_checkpointing'],
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

    print("\nTrainer ready!")
    print(f"Training config:")
    print(f"  - Device: {gpu_config['device']}")
    print(f"  - Batch size: {gpu_config['batch_size']}")
    print(f"  - Max length: {gpu_config['max_length']}")
    print(f"  - FP16: {gpu_config['use_fp16']}")
    print(f"  - Gradient checkpointing: {gpu_config['gradient_checkpointing']}")

    if RUN_TRAINING:
        print("\n>>> Running LoRA training (1 epoch)...")
        trainer.train()
        print(">>> LoRA training complete!")

        # Save LoRA adapters only (very small!)
        model.save_pretrained("./lora_adapters")
        print("\n>>> LoRA adapters saved to ./lora_adapters/")
        print(f"    Adapter size: Only the LoRA weights (~few MB)")
    else:
        print("\nTo train: trainer.train()")
        print("To save adapters: model.save_pretrained('./lora_adapters')")

    return model, tokenizer


# =============================================================================
# Part 5: QLoRA (Quantized LoRA) - For Large Models
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
    print("Part 5: QLoRA Setup (for Large Models)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\nQLoRA requires CUDA. Showing configuration only...")
        show_qlora_config()
        return None, None

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Only attempt to load large models if we have enough VRAM
    if vram_gb < 6:
        print(f"\nVRAM ({vram_gb:.1f}GB) too low for QLoRA demo.")
        print("QLoRA is for large models (7B+) that need 6GB+ VRAM.")
        show_qlora_config()
        return None, None

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    except ImportError:
        print("Install bitsandbytes: pip install bitsandbytes")
        show_qlora_config()
        return None, None

    # Use TinyLlama for demonstration (1.1B params)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"\nLoading {model_name} with 4-bit quantization...")
    print("(This would work similarly for Llama-2-7B, Mistral-7B, etc.)")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # Normalized float 4-bit
        bnb_4bit_compute_dtype=torch.float16, # Compute in fp16
        bnb_4bit_use_double_quant=True,      # Nested quantization
    )

    try:
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
    except Exception as e:
        print(f"\nCouldn't load model: {e}")
        print("This is normal if you have limited VRAM.")
        show_qlora_config()
        return None, None


def show_qlora_config():
    """Show QLoRA configuration without running it."""
    print("\nQLoRA Configuration Example:")
    config_code = '''
    # QLoRA enables fine-tuning 7B-70B models on consumer GPUs!

    from transformers import BitsAndBytesConfig, AutoModelForCausalLM
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

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

    # Memory comparison (Llama-2-7B):
    # - Full precision: ~28 GB
    # - 4-bit: ~4 GB
    # - 4-bit + LoRA training: ~6-8 GB ✓ Fits on RTX 3060!
    '''
    print(config_code)


# =============================================================================
# Part 6: Loading and Using LoRA Adapters
# =============================================================================

def using_lora_adapters():
    """
    How to save, load, and use LoRA adapters.
    """
    print("\n" + "=" * 60)
    print("Part 6: Using LoRA Adapters")
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

    print("\n2. Loading LoRA adapters onto base model:")
    print("   base_model = AutoModelForCausalLM.from_pretrained('distilgpt2')")
    print("   model = PeftModel.from_pretrained(base_model, './my_lora_adapters')")

    print("\n3. Merging adapters into base model (for deployment):")
    print("   merged_model = model.merge_and_unload()")
    print("   merged_model.save_pretrained('./merged_model')")
    print("   # Now it's a regular model, no PEFT dependency needed")

    print("\n4. Switching between adapters (multi-task):")
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
# Part 7: DPO + LoRA - Best of Both Worlds
# =============================================================================

def dpo_with_lora():
    """
    Combine DPO (Direct Preference Optimization) with LoRA.

    This is a powerful combination:
    - DPO: Simple alignment method, no reward model needed
    - LoRA: Memory-efficient, trains only 0.1-1% of parameters

    Use case: Align a model to preferences with minimal memory.
    """
    print("\n" + "=" * 60)
    print("Part 7: DPO + LoRA (Preference Alignment)")
    print("=" * 60)

    print("\nWhy DPO + LoRA?")
    print("  ✓ DPO is simpler than PPO (no reward model, no RL instability)")
    print("  ✓ LoRA makes it memory-efficient")
    print("  ✓ Perfect for aligning models on consumer GPUs")
    print("  ✓ Used in production by many companies")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from trl import DPOConfig, DPOTrainer
        from datasets import Dataset
    except ImportError as e:
        print(f"\nMissing package: {e}")
        print("Install: pip install trl")
        return None, None

    # Detect GPU config
    gpu_config = detect_gpu_vram()

    model_name = "distilgpt2"
    print(f"\nLoading model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    load_kwargs = {
        'pretrained_model_name_or_path': model_name,
        'torch_dtype': torch.float16 if gpu_config['use_fp16'] else torch.float32,
    }

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    # Reference model (frozen) - DPO needs this for KL penalty
    ref_model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    print(f"Model parameters: {model.num_parameters():,}")

    # Apply LoRA to the main model (not reference model)
    lora_config = LoraConfig(
        r=16,                       # Higher rank for alignment
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nAfter LoRA:")
    print(f"  - Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Enable gradient checkpointing if needed
    if gpu_config['gradient_checkpointing']:
        model.enable_input_require_grads()
        if hasattr(model.base_model, 'gradient_checkpointing_enable'):
            model.base_model.gradient_checkpointing_enable()

    # Create preference dataset
    # Format: prompt, chosen (better response), rejected (worse response)
    print("\nCreating preference dataset...")

    preference_data = [
        {
            "prompt": "Explain what Python is.",
            "chosen": "Python is a high-level, interpreted programming language known for its clear syntax and readability. It supports multiple programming paradigms and has extensive libraries for various applications.",
            "rejected": "python is a programming language i guess",
        },
        {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a subset of artificial intelligence where systems learn patterns from data to make predictions or decisions without being explicitly programmed for each specific task.",
            "rejected": "ml is when computers do stuff automatically",
        },
        {
            "prompt": "Explain how a computer works.",
            "chosen": "A computer processes data using a CPU (central processing unit) that executes instructions stored in memory. It takes input from devices, processes it according to programs, and produces output.",
            "rejected": "computers work by doing calculations fast",
        },
        {
            "prompt": "What is an API?",
            "chosen": "An API (Application Programming Interface) is a set of protocols and tools that allows different software applications to communicate with each other. It defines methods for requesting and exchanging data.",
            "rejected": "api is how programs talk i think",
        },
    ] * 10  # 40 examples

    dataset = Dataset.from_list(preference_data)

    print(f"Dataset size: {len(dataset)} preference pairs")
    print("\nExample preference pair:")
    print(f"  Prompt: {preference_data[0]['prompt']}")
    print(f"  Chosen: {preference_data[0]['chosen'][:60]}...")
    print(f"  Rejected: {preference_data[0]['rejected'][:60]}...")

    # DPO Config
    dpo_config = DPOConfig(
        output_dir="./dpo_lora_output",
        num_train_epochs=1,
        per_device_train_batch_size=gpu_config['batch_size'],
        gradient_accumulation_steps=2,
        learning_rate=5e-5,  # Lower LR for DPO
        beta=0.1,  # DPO temperature (controls strength of preference)
        max_length=gpu_config['max_length'],
        max_prompt_length=gpu_config['max_length'] // 2,
        logging_steps=5,
        fp16=gpu_config['use_fp16'],
        report_to="none",
        remove_unused_columns=False,
    )

    print(f"\nDPO Config:")
    print(f"  - Beta (temperature): {dpo_config.beta}")
    print(f"  - Learning rate: {dpo_config.learning_rate}")
    print(f"  - Batch size: {dpo_config.per_device_train_batch_size}")

    # Create DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nDPOTrainer with LoRA ready!")

    print("\nHow DPO works:")
    print("  1. For each (prompt, chosen, rejected) pair:")
    print("  2. Compute log probabilities: log π(chosen) and log π(rejected)")
    print("  3. Loss = -log σ(β * [log π(chosen) - log π(rejected)])")
    print("  4. This directly optimizes the policy from preferences!")
    print("  5. No reward model needed, more stable than PPO")

    if RUN_TRAINING:
        print("\n>>> Running DPO + LoRA training (1 epoch)...")
        trainer.train()
        print(">>> DPO + LoRA training complete!")

        # Save LoRA adapters
        model.save_pretrained("./dpo_lora_adapters")
        print("\n>>> DPO-aligned LoRA adapters saved to ./dpo_lora_adapters/")
    else:
        print("\nTo train: trainer.train()")
        print("To save: model.save_pretrained('./dpo_lora_adapters')")

    print("\n" + "-" * 40)
    print("Use cases for DPO + LoRA:")
    print("  • Align models to be more helpful/harmless")
    print("  • Improve response quality with preference data")
    print("  • Style transfer (formal vs casual)")
    print("  • Domain adaptation with expert preferences")
    print("  • Fine-tune on consumer GPU (4-8GB VRAM)")

    return model, tokenizer


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the LoRA tutorial."""
    global RUN_TRAINING

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="LoRA Fine-Tuning Tutorial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lora_finetuning.py           # Show tutorial (no training)
  python lora_finetuning.py --train   # Actually run training (1 epoch)
        """
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Actually run training (1 epoch). Without this flag, only shows setup."
    )
    args = parser.parse_args()

    RUN_TRAINING = args.train

    print("=" * 60)
    print("LoRA Fine-Tuning Tutorial")
    print("=" * 60)

    if RUN_TRAINING:
        print("\n*** TRAINING MODE: Will run actual training (1 epoch) ***\n")
    else:
        print("\n*** DEMO MODE: Showing setup only. Use --train to run training ***\n")

    if not check_dependencies():
        return

    try:
        # Run tutorial sections
        explain_lora_config()
        model, tokenizer = train_with_lora()
        setup_qlora()
        using_lora_adapters()
        dpo_with_lora()

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
    1. LoRA trains only 0.1-1% of parameters
    2. Much faster and uses less memory than full fine-tuning
    3. Automatic GPU detection and VRAM optimization
    4. QLoRA adds 4-bit quantization for large models
    5. Adapters are small (~few MB) and easy to share
    6. Can merge adapters back into base model for deployment
    7. DPO + LoRA combines preference optimization with efficiency

    What You Learned:
    ----------------
    • Basic LoRA: Parameter-efficient fine-tuning
    • QLoRA: 4-bit quantization for 7B+ models
    • DPO + LoRA: Preference alignment with minimal memory
    • Adapter management: Save, load, merge, multi-task

    Memory Requirements:
    -------------------
    - CPU/No GPU: Works! (slower, uses synthetic data)
    - <4GB VRAM: 8-bit quantization + small batches
    - 4-8GB VRAM: FP16 + gradient checkpointing
    - 8-12GB VRAM: Standard LoRA training
    - 12GB+ VRAM: Can train larger models with QLoRA

    Next steps:
    - Try: python lora_finetuning.py --train
    - Use your own dataset from HuggingFace
    - Experiment with different ranks (r) and alpha values
    - Try DPO + LoRA for alignment tasks
    - Try QLoRA for 7B+ models if you have 8GB+ VRAM
    """)


if __name__ == "__main__":
    main()
