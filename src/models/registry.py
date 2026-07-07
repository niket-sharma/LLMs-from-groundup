"""
Model registry: config -> model factory.

Every module in the curriculum builds models through ``create_model`` so that
architectural variants (RoPE, RMSNorm, GQA, ...) are selected by config, not
by importing different classes. As modules M1/M2 land their implementations,
the SUPPORTED_* sets below grow.
"""

from typing import Union, Dict, Any

from .config import GPTConfig, PRESETS
from .gpt import SmallGPT

# What the current model implementation actually supports. Config values
# outside these sets are valid *configurations* (they can be serialized,
# compared, planned around) but cannot be instantiated yet.
SUPPORTED = {
    "pos_encoding": {"learned"},    # + "sinusoidal", "rope" in M1.2
    "norm": {"layernorm"},          # + "rmsnorm" in M1.3
    "activation": {"gelu"},         # + "swiglu" in M1.3
    "attention": {"mha"},           # + "mqa", "gqa" in M2.2
}


def create_model(config: Union[GPTConfig, Dict[str, Any], str, None] = None) -> SmallGPT:
    """
    Build a model from a GPTConfig, a preset name, or a plain dict.

    Examples:
        create_model()                          # historical SmallGPT defaults
        create_model("tiny")                    # preset by name
        create_model(GPTConfig(preset="base"))  # explicit config
        create_model({"d_model": 128, "n_heads": 4})
    """
    if config is None:
        config = GPTConfig()
    elif isinstance(config, str):
        config = GPTConfig(preset=config)
    elif isinstance(config, dict):
        config = GPTConfig.from_dict(config)

    for knob, supported in SUPPORTED.items():
        value = getattr(config, knob)
        if value not in supported:
            raise NotImplementedError(
                f"{knob}={value!r} is planned but not implemented yet "
                f"(currently supported: {sorted(supported)}). "
                f"See llm-engineer-enhancement.md for the module that adds it."
            )

    model = SmallGPT(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    )
    # Attach the config so checkpointing/benchmarks can round-trip it.
    model.config = config
    return model


def list_presets() -> Dict[str, Dict[str, int]]:
    """Return the available preset architectures."""
    return {name: dict(dims) for name, dims in PRESETS.items()}
