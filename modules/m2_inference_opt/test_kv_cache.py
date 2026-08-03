"""Parity and allocation tests for Module 2.1 KV caching."""

import torch

from models import GPTConfig, create_model
from utils.inference import GPTInference
from training.dataset import SimpleTokenizer

from .kv_cache import DynamicKVCache, StaticKVCache, kv_cache_bytes_per_token


def tiny_model():
    torch.manual_seed(7)
    return create_model(GPTConfig(
        preset="tiny", vocab_size=41, max_seq_len=64, dropout=0.0, pos_encoding="rope"
    )).eval()


def test_cached_logits_match_full_forward():
    model = tiny_model()
    tokens = torch.randint(0, 41, (2, 17))
    with torch.no_grad():
        full = model(tokens)
        past = None
        pieces = []
        for index in range(tokens.size(1)):
            logits, past = model(tokens[:, index:index + 1], past_key_values=past, use_cache=True)
            pieces.append(logits)
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=1e-5, rtol=1e-5)


def test_cached_greedy_generation_is_token_identical():
    model = tiny_model()
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab("abcdefghijklmnopqrstuvwxyz ")
    # The tokenizer's vocabulary is smaller than the model's, which is valid for
    # generation and keeps this test close to the public GPTInference API.
    prompt = "hello"
    inference = GPTInference(model, tokenizer, device="cpu")
    naive = inference.generate(prompt, max_new_tokens=12, do_sample=False)
    cached = inference.generate(prompt, max_new_tokens=12, do_sample=False, use_kv_cache=True)
    assert cached == naive


def test_dynamic_and_static_cache_match():
    key1, value1 = torch.randn(2, 4, 3, 8), torch.randn(2, 4, 3, 8)
    key2, value2 = torch.randn(2, 4, 2, 8), torch.randn(2, 4, 2, 8)
    dynamic = DynamicKVCache()
    dynamic.append(key1, value1)
    dynamic_k, dynamic_v = dynamic.append(key2, value2)
    static = StaticKVCache(2, 4, 8, 8, dtype=key1.dtype)
    static.append(key1, value1)
    static_k, static_v = static.append(key2, value2)
    assert torch.equal(dynamic_k, static_k)
    assert torch.equal(dynamic_v, static_v)
    assert static.length == 5


def test_kv_cache_memory_formula():
    # 4 layers × K/V × 4 heads × 32 dims × 4 fp32 bytes.
    assert kv_cache_bytes_per_token(4, 4, 32) == 4 * 2 * 4 * 32 * 4

