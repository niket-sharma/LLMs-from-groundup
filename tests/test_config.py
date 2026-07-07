import pytest
import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.config import GPTConfig, PRESETS
from models.registry import create_model, list_presets
from utils.helpers import count_params, estimate_flops


class TestGPTConfig:
    def test_default_config_matches_smallgpt_defaults(self):
        config = GPTConfig()
        assert config.d_model == 384
        assert config.n_heads == 6
        assert config.n_layers == 6
        assert config.d_ff == 1536
        assert config.max_seq_len == 1024
        assert config.vocab_size == 50257

    def test_all_presets_valid(self):
        for name in PRESETS:
            config = GPTConfig(preset=name)
            assert config.d_model % config.n_heads == 0

    def test_preset_override(self):
        config = GPTConfig(preset="tiny", max_seq_len=64, vocab_size=100)
        assert config.d_model == PRESETS["tiny"]["d_model"]
        assert config.max_seq_len == 64
        assert config.vocab_size == 100

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="preset"):
            GPTConfig(preset="mega")

    def test_d_ff_defaults_to_4x(self):
        config = GPTConfig(d_model=128, n_heads=4)
        assert config.d_ff == 512

    def test_invalid_head_split_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            GPTConfig(d_model=100, n_heads=6)

    def test_invalid_knob_values_raise(self):
        with pytest.raises(ValueError):
            GPTConfig(pos_encoding="alibi")
        with pytest.raises(ValueError):
            GPTConfig(norm="batchnorm")
        with pytest.raises(ValueError):
            GPTConfig(activation="relu6")
        with pytest.raises(ValueError):
            GPTConfig(attention="mla")

    def test_kv_head_normalization(self):
        assert GPTConfig(attention="mha").n_kv_heads == 6
        assert GPTConfig(attention="mqa").n_kv_heads == 1
        assert GPTConfig(attention="gqa", n_kv_heads=2).n_kv_heads == 2
        with pytest.raises(ValueError):
            GPTConfig(attention="gqa")  # gqa requires n_kv_heads
        with pytest.raises(ValueError):
            GPTConfig(attention="gqa", n_kv_heads=4)  # 6 % 4 != 0

    def test_dict_round_trip(self):
        config = GPTConfig(preset="tiny", vocab_size=1000)
        restored = GPTConfig.from_dict(config.to_dict())
        assert restored == config


class TestCreateModel:
    def test_create_from_config(self):
        model = create_model(GPTConfig(preset="tiny", vocab_size=500))
        tokens = torch.randint(0, 500, (2, 8))
        logits = model(tokens)
        assert logits.shape == (2, 8, 500)
        assert model.config.preset == "tiny"

    def test_create_from_preset_name(self):
        model = create_model("tiny")
        assert model.d_model == PRESETS["tiny"]["d_model"]

    def test_create_from_dict(self):
        model = create_model({"d_model": 128, "n_heads": 4, "n_layers": 2, "vocab_size": 100})
        assert model.d_model == 128
        assert model.n_layers == 2

    def test_create_default(self):
        model = create_model()
        assert model.d_model == 384

    def test_unimplemented_knob_raises(self):
        for kwargs in (
            {"pos_encoding": "rope"},
            {"norm": "rmsnorm"},
            {"activation": "swiglu"},
            {"attention": "gqa", "n_kv_heads": 2},
        ):
            with pytest.raises(NotImplementedError):
                create_model(GPTConfig(preset="tiny", **kwargs))

    def test_preset_sizes_ordered(self):
        sizes = {}
        for name in ("tiny", "small"):
            model = create_model(GPTConfig(preset=name, vocab_size=1000))
            sizes[name] = count_params(model, non_embedding=True)
        assert sizes["tiny"] < sizes["small"]
        # tiny should be around ~1M non-embedding params
        assert sizes["tiny"] < 2_000_000

    def test_list_presets(self):
        presets = list_presets()
        assert set(presets) == {"tiny", "small", "base", "gpt2-ish"}


class TestHelpers:
    def test_count_params_non_embedding(self):
        model = create_model(GPTConfig(preset="tiny", vocab_size=1000))
        total = count_params(model)
        non_emb = count_params(model, non_embedding=True)
        assert 0 < non_emb < total

    def test_estimate_flops(self):
        model = create_model(GPTConfig(preset="tiny", vocab_size=1000))
        est = estimate_flops(model, seq_len=128)
        assert est['flops_per_token_fwd'] > 2 * est['params_non_embedding']
        assert est['flops_per_seq_fwd'] == est['flops_per_token_fwd'] * 128
        assert est['flops_per_token_train'] == 3 * est['flops_per_token_fwd']
