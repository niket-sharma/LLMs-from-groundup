import pytest
import torch
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.inference import GPTInference, generate_text
from utils.helpers import count_parameters, set_seed, get_device, print_model_info
from models.gpt import SmallGPT
from training.dataset import SimpleTokenizer


class TestInference:
    def test_gpt_inference_init(self):
        model = SmallGPT(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab("abcdefghijklmnopqrstuvwxyz ")

        inference = GPTInference(model, tokenizer)

        assert inference.model is not None
        assert inference.tokenizer is not None
        assert inference.device in ['cpu', 'cuda', 'mps']

    def test_gpt_inference_generate(self):
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab("abcdefghijklmnopqrstuvwxyz ")
        model = SmallGPT(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=2)

        inference = GPTInference(model, tokenizer, device='cpu')

        prompt = "hello"
        generated = inference.generate(prompt, max_new_tokens=5, temperature=1.0)

        assert isinstance(generated, str)
        assert len(generated) >= len(prompt)

    def test_generate_text_function(self):
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab("abcdefghijklmnopqrstuvwxyz ")
        model = SmallGPT(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=2)

        prompt = "test"
        generated = generate_text(
            model, tokenizer, prompt, max_new_tokens=5, device='cpu'
        )

        assert isinstance(generated, str)
        assert len(generated) >= len(prompt)

    def test_inference_batch_generation(self):
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab("abcdefghijklmnopqrstuvwxyz ")
        model = SmallGPT(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=2)

        inference = GPTInference(model, tokenizer, device='cpu')

        prompts = ["hello", "world", "test"]
        generated = inference.generate_batch(prompts, max_new_tokens=3)

        assert len(generated) == len(prompts)
        assert all(isinstance(text, str) for text in generated)


class TestHelpers:
    def test_count_parameters(self):
        model = SmallGPT(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        param_count = count_parameters(model)

        assert isinstance(param_count, int)
        assert param_count > 0

    def test_set_seed(self):
        set_seed(42)

        # Generate some random numbers
        torch_rand1 = torch.rand(5)
        set_seed(42)
        torch_rand2 = torch.rand(5)

        # Should be the same due to seeding
        assert torch.allclose(torch_rand1, torch_rand2)

    def test_get_device(self):
        device = get_device()
        assert device in ['cpu', 'cuda', 'mps']

    def test_print_model_info(self, capsys):
        model = SmallGPT(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        print_model_info(model)

        captured = capsys.readouterr()
        assert "trainable parameters" in captured.out
        assert "Model size" in captured.out