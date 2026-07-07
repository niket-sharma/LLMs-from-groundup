"""
Config system for the GPT model family.

GPTConfig is the single source of truth for every architectural knob in the
repo. Later modules (M1: RoPE/RMSNorm/SwiGLU, M2: GQA/KV-cache/flash) add
implementations behind these fields; the fields exist now so the factory API
never changes.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


# Valid values for each architectural choice. Options marked "planned" are
# accepted by the config (so experiments can be described up front) but the
# model factory raises NotImplementedError until the owning module lands.
POS_ENCODINGS = ("learned", "sinusoidal", "rope")   # rope/sinusoidal: M1.2
NORMS = ("layernorm", "rmsnorm")                    # rmsnorm: M1.3
ACTIVATIONS = ("gelu", "swiglu")                    # swiglu: M1.3
ATTENTION_KINDS = ("mha", "mqa", "gqa")             # mqa/gqa: M2.2


# Preset architecture dimensions. Approximate non-embedding parameter counts
# (~12 * n_layers * d_model^2) are noted; total size depends on vocab_size.
PRESETS: Dict[str, Dict[str, int]] = {
    # ~0.8M non-embedding params
    "tiny": dict(d_model=128, n_heads=4, n_layers=4, d_ff=512, max_seq_len=256),
    # ~10.6M non-embedding params (matches the original SmallGPT defaults)
    "small": dict(d_model=384, n_heads=6, n_layers=6, d_ff=1536, max_seq_len=512),
    # ~49M non-embedding params
    "base": dict(d_model=640, n_heads=10, n_layers=10, d_ff=2560, max_seq_len=1024),
    # GPT-2 small shape: ~124M total with the 50257 vocab (weight-tied)
    "gpt2-ish": dict(d_model=768, n_heads=12, n_layers=12, d_ff=3072, max_seq_len=1024),
}


@dataclass
class GPTConfig:
    """
    Configuration for a GPT-style model.

    Either pass a ``preset`` name ("tiny", "small", "base", "gpt2-ish") or set
    the size fields explicitly. Explicitly-passed size fields override the
    preset's values.

    Architectural knobs default to the classic GPT-2 recipe (learned
    positional embeddings, LayerNorm, GELU, multi-head attention).
    """

    preset: Optional[str] = None

    # Size / shape
    vocab_size: int = 50257
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    n_layers: Optional[int] = None
    d_ff: Optional[int] = None
    max_seq_len: Optional[int] = None
    dropout: float = 0.1

    # Architectural knobs (implemented incrementally by modules M1/M2)
    pos_encoding: str = "learned"       # learned | sinusoidal | rope
    norm: str = "layernorm"             # layernorm | rmsnorm
    activation: str = "gelu"            # gelu | swiglu
    attention: str = "mha"              # mha | mqa | gqa
    n_kv_heads: Optional[int] = None    # for gqa: kv heads; mqa forces 1
    use_kv_cache: bool = False          # M2.1
    use_flash: bool = False             # M2.3 (F.scaled_dot_product_attention)
    tie_weights: bool = True

    def __post_init__(self):
        if self.preset is not None:
            if self.preset not in PRESETS:
                raise ValueError(
                    f"Unknown preset {self.preset!r}. Available: {sorted(PRESETS)}"
                )
            for key, value in PRESETS[self.preset].items():
                if getattr(self, key) is None:
                    setattr(self, key, value)

        # Fall back to the historical SmallGPT defaults for anything unset.
        defaults = dict(d_model=384, n_heads=6, n_layers=6, max_seq_len=1024)
        for key, value in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

        self._validate()

    def _validate(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.pos_encoding not in POS_ENCODINGS:
            raise ValueError(f"pos_encoding must be one of {POS_ENCODINGS}")
        if self.norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if self.attention not in ATTENTION_KINDS:
            raise ValueError(f"attention must be one of {ATTENTION_KINDS}")

        # Normalize n_kv_heads per attention kind.
        if self.attention == "mha":
            if self.n_kv_heads is not None and self.n_kv_heads != self.n_heads:
                raise ValueError("mha requires n_kv_heads == n_heads (or leave unset)")
            self.n_kv_heads = self.n_heads
        elif self.attention == "mqa":
            if self.n_kv_heads not in (None, 1):
                raise ValueError("mqa requires n_kv_heads == 1 (or leave unset)")
            self.n_kv_heads = 1
        elif self.attention == "gqa":
            if self.n_kv_heads is None:
                raise ValueError("gqa requires n_kv_heads to be set")
            if self.n_heads % self.n_kv_heads != 0:
                raise ValueError(
                    f"n_heads ({self.n_heads}) must be divisible by "
                    f"n_kv_heads ({self.n_kv_heads})"
                )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GPTConfig":
        return cls(**d)
