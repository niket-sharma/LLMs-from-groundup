# Contributing to Small GPT

Thank you for your interest in contributing to the Small GPT project! This document provides guidelines for contributing to this educational implementation of a GPT-style language model.

## 🎯 Project Goals

This project aims to provide:
- A clear, educational implementation of transformer architecture
- Well-documented, readable code for learning purposes
- Comprehensive tests and examples
- Modular design that's easy to extend and experiment with

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/LLMs-from-groundup.git
   cd LLMs-from-groundup
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```
4. Run tests to ensure everything works:
   ```bash
   pytest tests/ -v
   ```

## 🔧 Development Setup

### Code Style
We use the following tools for code consistency:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Format your code before submitting:
```bash
black src/ tests/ examples/
isort src/ tests/ examples/
flake8 src/ tests/ examples/
```

### Testing
- Write tests for new features
- Ensure all tests pass before submitting
- Add docstrings to new functions and classes
- Test coverage should be maintained or improved

## 📝 Contribution Types

### 🐛 Bug Reports
When reporting bugs, please include:
- Python version and operating system
- PyTorch version
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages (if any)

### ✨ Feature Requests
For new features, please:
- Describe the feature and its purpose
- Explain how it fits with the educational goals
- Provide example usage if possible
- Consider backward compatibility

### 🔍 Code Contributions

#### Areas for Contribution
- **Model Architecture**: New attention mechanisms, normalization techniques
- **Training**: Advanced optimizers, learning rate schedules, regularization
- **Data**: Better tokenizers, data augmentation, preprocessing
- **Inference**: New sampling strategies, beam search, batched inference
- **Examples**: Tutorial notebooks, training on specific datasets
- **Documentation**: Improve explanations, add theory background
- **Testing**: Edge cases, performance tests, integration tests

#### Pull Request Process
1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add/update tests
4. Update documentation
5. Format your code
6. Run tests:
   ```bash
   pytest tests/ -v
   ```
7. Commit your changes:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```
8. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
9. Create a Pull Request

### 📚 Documentation Contributions
- Fix typos or unclear explanations
- Add examples for existing features
- Improve code comments
- Create tutorials or guides
- Add theoretical background explanations

## 🎨 Code Style Guidelines

### Python Style
- Follow PEP 8
- Use type hints where possible
- Write clear, descriptive variable names
- Add docstrings to all public functions/classes

### Example:
```python
def attention_function(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    \"\"\"
    Compute scaled dot-product attention.

    Args:
        query: Query tensor of shape (batch, seq_len, d_model)
        key: Key tensor of shape (batch, seq_len, d_model)
        value: Value tensor of shape (batch, seq_len, d_model)
        mask: Optional attention mask

    Returns:
        attention_output: Attended values
        attention_weights: Attention weights for visualization
    \"\"\"
    # Implementation here
    pass
```

### Commit Messages
Use clear, descriptive commit messages:
- `feat: add top-p sampling to inference`
- `fix: resolve attention mask broadcasting issue`
- `docs: improve transformer block documentation`
- `test: add tests for positional encoding`

## 🧪 Testing Guidelines

### Test Structure
- Unit tests for individual components
- Integration tests for complete workflows
- Performance tests for critical paths

### Test Examples
```python
def test_attention_output_shape():
    \"\"\"Test that attention produces correct output shape.\"\"\"
    attention = MultiHeadAttention(d_model=384, n_heads=6)
    x = torch.randn(2, 10, 384)  # batch, seq, features

    output, weights = attention(x, x, x)

    assert output.shape == (2, 10, 384)
    assert weights.shape == (2, 6, 10, 10)
```

## 📋 Review Process

### What We Look For
- **Correctness**: Does the code work as intended?
- **Clarity**: Is the code easy to understand?
- **Testing**: Are there adequate tests?
- **Documentation**: Is the change well-documented?
- **Educational Value**: Does it help others learn?

### Review Timeline
- Initial response: Within 3-5 days
- Full review: Within 1-2 weeks
- We may request changes or provide feedback

## 🤝 Community Guidelines

### Be Respectful
- Use welcoming and inclusive language
- Respect different viewpoints and experiences
- Focus on what's best for the community
- Show empathy towards other members

### Be Constructive
- Provide helpful feedback
- Explain reasoning behind suggestions
- Help others learn and grow
- Share knowledge generously

## ❓ Questions?

If you have questions about contributing:
- Open an issue with the "question" label
- Check existing issues and discussions
- Read the project documentation

## 🎉 Recognition

Contributors will be recognized in:
- README acknowledgments
- Release notes for significant contributions
- Project documentation

Thank you for helping make this project better for everyone! 🚀