"""KV-cache utilities used by Module 2's incremental decoding experiments.

The model API uses a tuple of per-layer ``(K, V)`` tensors. This file makes the
two allocation strategies explicit so learners can inspect the memory tradeoff:
``DynamicKVCache`` appends tensors (simple, allocation-heavy) while
``StaticKVCache`` preallocates the maximum sequence length (predictable memory).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


KV = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class DynamicKVCache:
    """Append-only educational cache; appropriate for variable-length demos."""

    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None

    def append(self, key: torch.Tensor, value: torch.Tensor) -> KV:
        if self.key is None:
            self.key, self.value = key, value
        else:
            self.key = torch.cat((self.key, key), dim=2)
            self.value = torch.cat((self.value, value), dim=2)
        return self.key, self.value

    @property
    def length(self) -> int:
        return 0 if self.key is None else self.key.size(2)


class StaticKVCache:
    """Preallocated cache with O(1) writes and a fixed maximum length."""

    def __init__(self, batch_size: int, n_heads: int, max_seq_len: int, head_dim: int,
                 *, device=None, dtype=None):
        shape = (batch_size, n_heads, max_seq_len, head_dim)
        self.key = torch.empty(shape, device=device, dtype=dtype)
        self.value = torch.empty(shape, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len
        self.length = 0

    def append(self, key: torch.Tensor, value: torch.Tensor) -> KV:
        new_len = key.size(2)
        if self.length + new_len > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
        end = self.length + new_len
        self.key[:, :, self.length:end].copy_(key)
        self.value[:, :, self.length:end].copy_(value)
        self.length = end
        return self.key[:, :, :end], self.value[:, :, :end]


def kv_cache_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int,
                             dtype: torch.dtype = torch.float32) -> int:
    """Bytes/token for one batch item: layers × K/V × heads × head_dim × dtype."""
    return n_layers * 2 * n_kv_heads * head_dim * torch.empty((), dtype=dtype).element_size()

