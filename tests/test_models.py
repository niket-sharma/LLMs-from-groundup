import pytest
import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gpt import SmallGPT, create_small_gpt
from models.attention import MultiHeadAttention, CausalSelfAttention
from models.feedforward import FeedForward, TransformerBlock
from models.embeddings import TokenEmbedding, PositionalEncoding, GPTEmbedding


class TestAttention:
    def test_multihead_attention_init(self):
        d_model, n_heads = 384, 6
        attention = MultiHeadAttention(d_model, n_heads)
        assert attention.d_model == d_model
        assert attention.n_heads == n_heads
        assert attention.d_k == d_model // n_heads

    def test_multihead_attention_forward(self):
        d_model, n_heads = 384, 6
        batch_size, seq_len = 2, 10

        attention = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output, weights = attention(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len)

    def test_causal_self_attention(self):
        d_model, n_heads, max_seq_len = 384, 6, 1024
        batch_size, seq_len = 2, 10

        attention = CausalSelfAttention(d_model, n_heads, max_seq_len)
        x = torch.randn(batch_size, seq_len, d_model)

        output, weights = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len)


class TestFeedForward:
    def test_feedforward_init(self):
        d_model, d_ff = 384, 1536
        ff = FeedForward(d_model, d_ff)
        assert ff.linear1.in_features == d_model
        assert ff.linear1.out_features == d_ff
        assert ff.linear2.in_features == d_ff
        assert ff.linear2.out_features == d_model

    def test_feedforward_forward(self):
        d_model, d_ff = 384, 1536
        batch_size, seq_len = 2, 10

        ff = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ff(x)
        assert output.shape == (batch_size, seq_len, d_model)

    def test_transformer_block(self):
        d_model, n_heads, d_ff, max_seq_len = 384, 6, 1536, 1024
        batch_size, seq_len = 2, 10

        block = TransformerBlock(d_model, n_heads, d_ff, max_seq_len)
        x = torch.randn(batch_size, seq_len, d_model)

        output, weights = block(x)
        assert output.shape == (batch_size, seq_len, d_model)


class TestEmbeddings:
    def test_token_embedding(self):
        vocab_size, d_model = 50257, 384
        batch_size, seq_len = 2, 10

        embedding = TokenEmbedding(vocab_size, d_model)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = embedding(tokens)
        assert output.shape == (batch_size, seq_len, d_model)

    def test_positional_encoding(self):
        d_model, max_seq_len = 384, 1024
        seq_len = 10

        pe = PositionalEncoding(d_model, max_seq_len)
        x = torch.randn(seq_len, 1, d_model)

        output = pe(x)
        assert output.shape == (seq_len, 1, d_model)

    def test_gpt_embedding(self):
        vocab_size, d_model, max_seq_len = 50257, 384, 1024
        batch_size, seq_len = 2, 10

        embedding = GPTEmbedding(vocab_size, d_model, max_seq_len)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = embedding(tokens)
        assert output.shape == (batch_size, seq_len, d_model)


class TestSmallGPT:
    def test_model_init(self):
        model = SmallGPT()
        assert model.vocab_size == 50257
        assert model.d_model == 384
        assert model.n_heads == 6
        assert model.n_layers == 6

    def test_model_forward(self):
        model = SmallGPT()
        batch_size, seq_len = 2, 10

        tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len))

        # Test without targets (inference)
        logits = model(tokens)
        assert logits.shape == (batch_size, seq_len, model.vocab_size)

        # Test with targets (training)
        targets = torch.randint(0, model.vocab_size, (batch_size, seq_len))
        logits, loss = model(tokens, targets)
        assert logits.shape == (batch_size, seq_len, model.vocab_size)
        assert isinstance(loss.item(), float)

    def test_model_generation(self):
        model = SmallGPT()
        batch_size, seq_len = 1, 5
        max_new_tokens = 10

        tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len))

        generated = model.generate(tokens, max_new_tokens)
        assert generated.shape == (batch_size, seq_len + max_new_tokens)

    def test_create_small_gpt(self):
        config = {
            'vocab_size': 1000,
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
        }

        model = create_small_gpt(config)
        assert model.vocab_size == 1000
        assert model.d_model == 128
        assert model.n_heads == 4
        assert model.n_layers == 2

    def test_parameter_count(self):
        model = SmallGPT()
        param_count = model.get_num_params()
        assert param_count > 0
        assert isinstance(param_count, int)