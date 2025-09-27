# Small GPT: LLMs from Ground Up

A minimal implementation of a GPT-style transformer language model built from scratch using PyTorch. This project provides a complete implementation including the model architecture, training utilities, inference capabilities, and comprehensive tests.

## 🚀 Features

- **Complete GPT Architecture**: Multi-head attention, feed-forward networks, layer normalization, positional embeddings
- **Training Pipeline**: Data loading, tokenization, training loop with validation
- **Text Generation**: Support for various sampling strategies (temperature, top-k, top-p)
- **Comprehensive Tests**: Unit tests for all components
- **Examples**: Training scripts, inference examples, and Jupyter notebook demo
- **Modular Design**: Clean, extensible codebase with separation of concerns

## 📁 Project Structure

```
LLMs-from-groundup/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention.py      # Multi-head attention mechanisms
│   │   ├── embeddings.py     # Token and positional embeddings
│   │   ├── feedforward.py    # Feed-forward networks and transformer blocks
│   │   └── gpt.py           # Main GPT model implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py       # Data loading and tokenization
│   │   └── trainer.py       # Training utilities and trainer class
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── inference.py     # Text generation and inference utilities
│   │   └── helpers.py       # Helper functions and utilities
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py       # Tests for model components
│   ├── test_training.py     # Tests for training utilities
│   └── test_utils.py        # Tests for utility functions
├── examples/
│   ├── train_example.py     # Training script example
│   ├── inference_example.py # Inference script example
│   └── jupyter_demo.ipynb   # Interactive Jupyter notebook demo
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md               # This file
```

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/LLMs-from-groundup.git
cd LLMs-from-groundup
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training a Model

```python
from src.models.gpt import create_small_gpt
from src.training.dataset import prepare_data, create_dataloaders
from src.training.trainer import GPTTrainer

# Prepare data
train_dataset, val_dataset, tokenizer = prepare_data('your_text_file.txt')
train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)

# Create model
config = {
    'vocab_size': tokenizer.vocab_size,
    'd_model': 384,
    'n_heads': 6,
    'n_layers': 6,
    'max_seq_len': 1024
}
model = create_small_gpt(config)

# Train
trainer = GPTTrainer(model, train_loader, val_loader)
trainer.train(epochs=10, save_dir='checkpoints')
```

### Generating Text

```python
from src.utils.inference import GPTInference

# Load trained model and tokenizer
inference = GPTInference(model, tokenizer)

# Generate text
generated = inference.generate(
    "Once upon a time",
    max_new_tokens=100,
    temperature=0.8
)
print(generated)
```

## 📚 Examples

### 1. Training Example
Run the complete training example:
```bash
python examples/train_example.py
```

### 2. Inference Example
Generate text with a trained model:
```bash
python examples/inference_example.py
```

### 3. Jupyter Demo
Open and run the interactive notebook:
```bash
jupyter notebook examples/jupyter_demo.ipynb
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
pytest tests/test_models.py -v      # Test model components
pytest tests/test_training.py -v    # Test training utilities
pytest tests/test_utils.py -v       # Test utility functions
```

## 🏗 Model Architecture

The Small GPT model implements the standard transformer decoder architecture:

- **Multi-Head Attention**: Self-attention mechanism with causal masking
- **Position-wise Feed-Forward**: Two-layer MLP with GELU activation
- **Layer Normalization**: Pre-norm architecture for stable training
- **Positional Embeddings**: Learnable position embeddings
- **Weight Tying**: Shared weights between input and output embeddings

### Default Configuration

```python
{
    'vocab_size': 50257,      # Vocabulary size
    'd_model': 384,           # Model dimension
    'n_heads': 6,             # Number of attention heads
    'n_layers': 6,            # Number of transformer layers
    'd_ff': 1536,             # Feed-forward dimension
    'max_seq_len': 1024,      # Maximum sequence length
    'dropout': 0.1            # Dropout rate
}
```

## 🎯 Key Components

### 1. Attention Mechanism (`src/models/attention.py`)
- Multi-head self-attention
- Causal masking for autoregressive generation
- Scaled dot-product attention

### 2. Transformer Block (`src/models/feedforward.py`)
- Self-attention + feed-forward
- Residual connections
- Layer normalization

### 3. Embeddings (`src/models/embeddings.py`)
- Token embeddings with scaling
- Learnable positional embeddings
- Dropout for regularization

### 4. Training (`src/training/`)
- Simple character-level tokenizer
- DataLoader with proper batching
- Training loop with validation
- Gradient clipping and learning rate scheduling

### 5. Inference (`src/utils/inference.py`)
- Text generation with various sampling strategies
- Temperature scaling
- Top-k and top-p filtering
- Batch generation support

## 🔧 Customization

### Creating Custom Model Configurations

```python
# Tiny model for experimentation
tiny_config = {
    'vocab_size': 1000,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 4,
    'd_ff': 512,
    'max_seq_len': 256
}

# Larger model for better performance
large_config = {
    'vocab_size': 50257,
    'd_model': 768,
    'n_heads': 12,
    'n_layers': 12,
    'd_ff': 3072,
    'max_seq_len': 2048
}
```

### Custom Tokenizers

You can replace the simple character-level tokenizer with more sophisticated options:
- BPE (Byte-Pair Encoding)
- SentencePiece
- Hugging Face tokenizers

## 📊 Performance Tips

1. **Start Small**: Begin with a tiny model to verify your setup
2. **Monitor Loss**: Watch both training and validation loss
3. **Gradient Clipping**: Helps with training stability
4. **Learning Rate**: Start with 1e-3 and adjust based on loss curves
5. **Sequence Length**: Longer sequences require more memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📖 Educational Purpose

This implementation is designed for educational purposes to understand:
- Transformer architecture from first principles
- Self-attention mechanisms
- Autoregressive language modeling
- Training procedures for language models
- Text generation techniques

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Attention Is All You Need (Vaswani et al., 2017)
- Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
- PyTorch documentation and tutorials
- The transformer community for open research

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Learning! 🚀**