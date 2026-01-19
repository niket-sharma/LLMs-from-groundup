#!/usr/bin/env python3
"""
HuggingFace Local Inference Examples
====================================

This tutorial demonstrates how to run LLMs locally using HuggingFace:
1. Loading models and tokenizers
2. Text generation basics
3. Different generation strategies (greedy, sampling, beam search)
4. Chat templates for instruction-tuned models
5. Batched inference
6. Memory optimization (quantization, device_map)
7. Using pipelines for simplicity

Prerequisites:
    pip install transformers torch accelerate

For quantization (optional):
    pip install bitsandbytes
"""

import torch
from typing import List, Optional


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers torch accelerate")
        return False


# =============================================================================
# Example 1: Basic Model Loading and Generation
# =============================================================================

def basic_generation():
    """
    Load a model and generate text.

    This shows the fundamental workflow:
    1. Load tokenizer (text → tokens)
    2. Load model (tokens → predictions)
    3. Encode input → Generate → Decode output
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Model Loading and Generation")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Use a small model for demonstration
    model_name = "gpt2"  # ~500MB, runs on CPU

    print(f"\nLoading model: {model_name}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set pad token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded! Parameters: {model.num_parameters():,}")

    # Prepare input
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")

    print(f"\nPrompt: {prompt}")
    print(f"Input tokens: {inputs['input_ids'].tolist()}")

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated: {generated_text}")


# =============================================================================
# Example 2: Generation Strategies
# =============================================================================

def generation_strategies():
    """
    Compare different text generation strategies.

    - Greedy: Always pick most likely token (deterministic)
    - Sampling: Sample from probability distribution (random)
    - Top-k: Sample from top k most likely tokens
    - Top-p (nucleus): Sample from tokens with cumulative prob < p
    - Beam search: Keep top n sequences at each step
    """
    print("\n" + "=" * 60)
    print("Example 2: Generation Strategies")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt")

    strategies = {
        "Greedy (deterministic)": {
            "do_sample": False,
        },
        "Temperature=0.5 (focused)": {
            "do_sample": True,
            "temperature": 0.5,
        },
        "Temperature=1.5 (creative)": {
            "do_sample": True,
            "temperature": 1.5,
        },
        "Top-k=50": {
            "do_sample": True,
            "top_k": 50,
        },
        "Top-p=0.9 (nucleus)": {
            "do_sample": True,
            "top_p": 0.9,
        },
        "Beam search (n=5)": {
            "do_sample": False,
            "num_beams": 5,
        },
    }

    print(f"\nPrompt: {prompt}\n")

    for name, params in strategies.items():
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                pad_token_id=tokenizer.eos_token_id,
                **params
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"{name}:")
        print(f"  {text}\n")


# =============================================================================
# Example 3: Chat Models and Templates
# =============================================================================

def chat_models():
    """
    Use instruction-tuned chat models.

    Chat models expect a specific format (chat template).
    HuggingFace tokenizers handle this automatically.
    """
    print("\n" + "=" * 60)
    print("Example 3: Chat Models and Templates")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Use a small instruction-tuned model
    # Options: TinyLlama, Phi-2, Mistral-7B, Llama-2-7B-chat, etc.
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"\nLoading chat model: {model_name}")
    print("(This may take a moment for first download...)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Define conversation
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to check if a number is prime."},
    ]

    # Apply chat template
    # This converts messages to the format the model expects
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"\nFormatted prompt:\n{prompt[:200]}...")

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\nAssistant: {response}")


# =============================================================================
# Example 4: Batched Inference
# =============================================================================

def batched_inference():
    """
    Process multiple inputs efficiently in a batch.

    Batching is crucial for throughput in production.
    """
    print("\n" + "=" * 60)
    print("Example 4: Batched Inference")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set padding token and side
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for generation!

    prompts = [
        "The capital of France is",
        "Machine learning is",
        "Python programming",
        "The best way to learn",
    ]

    print(f"\nProcessing {len(prompts)} prompts in a batch...")

    # Tokenize with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode each output
    print("\nResults:")
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"{i+1}. {text}")


# =============================================================================
# Example 5: Memory Optimization with Quantization
# =============================================================================

def quantized_inference():
    """
    Load large models with quantization to reduce memory.

    4-bit quantization can reduce memory by ~4x with minimal quality loss.
    Requires: pip install bitsandbytes
    """
    print("\n" + "=" * 60)
    print("Example 5: Quantized Inference (Memory Optimization)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\nQuantization requires CUDA. Skipping...")
        return

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError:
        print("Install bitsandbytes: pip install bitsandbytes")
        return

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Nested quantization
        bnb_4bit_quant_type="nf4",  # Normalized float 4-bit
    )

    print(f"\nLoading {model_name} with 4-bit quantization...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # Check memory usage
    memory_mb = model.get_memory_footprint() / 1024 / 1024
    print(f"Model memory footprint: {memory_mb:.1f} MB")

    # Generate
    messages = [{"role": "user", "content": "What is quantum computing?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\nResponse: {response}")


# =============================================================================
# Example 6: Using Pipelines (Simplest API)
# =============================================================================

def pipeline_examples():
    """
    Use HuggingFace pipelines for the simplest API.

    Pipelines handle tokenization, inference, and decoding automatically.
    """
    print("\n" + "=" * 60)
    print("Example 6: Using Pipelines (Simplest API)")
    print("=" * 60)

    from transformers import pipeline

    # Text Generation Pipeline
    print("\n--- Text Generation ---")
    generator = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if torch.cuda.is_available() else -1,
    )

    result = generator(
        "The secret to happiness is",
        max_new_tokens=30,
        num_return_sequences=2,
        do_sample=True,
    )

    for i, r in enumerate(result):
        print(f"{i+1}. {r['generated_text']}")

    # Sentiment Analysis Pipeline
    print("\n--- Sentiment Analysis ---")
    classifier = pipeline("sentiment-analysis")

    texts = [
        "I love this product! It's amazing!",
        "This is the worst experience ever.",
        "It's okay, nothing special.",
    ]

    results = classifier(texts)
    for text, result in zip(texts, results):
        print(f"'{text[:30]}...' -> {result['label']} ({result['score']:.2f})")

    # Question Answering Pipeline
    print("\n--- Question Answering ---")
    qa = pipeline("question-answering")

    context = """
    Python is a high-level programming language created by Guido van Rossum.
    It was first released in 1991 and emphasizes code readability.
    Python supports multiple programming paradigms including procedural,
    object-oriented, and functional programming.
    """

    questions = [
        "Who created Python?",
        "When was Python first released?",
    ]

    for q in questions:
        result = qa(question=q, context=context)
        print(f"Q: {q}")
        print(f"A: {result['answer']} (confidence: {result['score']:.2f})")


# =============================================================================
# Example 7: Embeddings for Semantic Search
# =============================================================================

def embeddings_example():
    """
    Generate embeddings for semantic similarity and search.
    """
    print("\n" + "=" * 60)
    print("Example 7: Text Embeddings")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F

    # Use a sentence embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print(f"\nLoading embedding model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def get_embeddings(texts: List[str]) -> torch.Tensor:
        """Generate embeddings with mean pooling."""
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling
        attention_mask = inputs['attention_mask']
        embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        pooled = torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

        # Normalize
        return F.normalize(pooled, p=2, dim=1)

    # Example texts
    texts = [
        "I love programming in Python",
        "Python is my favorite language for coding",
        "The weather is beautiful today",
        "Machine learning is fascinating",
    ]

    embeddings = get_embeddings(texts)
    print(f"\nEmbedding shape: {embeddings.shape}")

    # Compute similarities
    similarities = torch.mm(embeddings, embeddings.T)

    print("\nCosine Similarity Matrix:")
    print("(1.0 = identical, 0.0 = unrelated)")
    print()

    # Print as table
    for i, text in enumerate(texts):
        print(f"Text {i+1}: '{text[:35]}...'")

    print("\n     ", end="")
    for i in range(len(texts)):
        print(f"  T{i+1}  ", end="")
    print()

    for i in range(len(texts)):
        print(f"T{i+1}  ", end="")
        for j in range(len(texts)):
            print(f" {similarities[i][j]:.2f} ", end="")
        print()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples."""
    print("=" * 60)
    print("HuggingFace Local Inference Tutorial")
    print("=" * 60)

    if not check_dependencies():
        return

    try:
        basic_generation()
        generation_strategies()
        # chat_models()  # Uncomment to test (downloads ~2GB model)
        batched_inference()
        # quantized_inference()  # Requires CUDA + bitsandbytes
        pipeline_examples()
        embeddings_example()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
