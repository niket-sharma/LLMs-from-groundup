"""
GPT (Generative Pre-trained Transformer) model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .embeddings import GPTEmbedding
from .feedforward import TransformerBlock


class SmallGPT(nn.Module):
    """
    A small GPT-style language model.

    This implements a decoder-only transformer architecture similar to GPT-2,
    designed for educational purposes and experimentation.

    Architecture:
    - Token + Positional embeddings
    - N transformer blocks (each with causal self-attention + FFN)
    - Final layer normalization
    - Linear head for next-token prediction

    Args:
        vocab_size: Size of the vocabulary
        d_model: Model/embedding dimension
        n_heads: Number of attention heads per layer
        n_layers: Number of transformer blocks
        d_ff: Feed-forward network hidden dimension
        max_seq_len: Maximum sequence length the model can handle
        dropout: Dropout probability for regularization
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        d_ff: int = 1536,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Store hyperparameters
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        # Embeddings
        self.embedding = GPTEmbedding(vocab_size, d_model, max_seq_len, dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        # Final layer normalization
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection (language model head)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between embedding and output layer
        # This is a common technique that reduces parameters and improves performance
        self.embedding.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            targets: Optional target token indices for computing loss

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, None otherwise
        """
        # Get embeddings
        x = self.embedding(idx)  # (batch_size, seq_len, d_model)

        # Pass through transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy: (batch_size * seq_len, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,  # Ignore padding tokens
            )
            return logits, loss

        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            idx: Starting token indices of shape (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens

        Returns:
            Generated token indices of shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop sequence if it exceeds max length
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

            # Get predictions
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # Get last position

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get the number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embedding.token_embedding.weight.numel()
            n_params -= self.embedding.position_embedding.weight.numel()
        return n_params


def create_small_gpt(config: Optional[Dict[str, Any]] = None) -> SmallGPT:
    """
    Factory function to create a SmallGPT model.

    Args:
        config: Optional configuration dictionary with model hyperparameters

    Returns:
        Configured SmallGPT model
    """
    if config is None:
        config = {}

    return SmallGPT(
        vocab_size=config.get('vocab_size', 50257),
        d_model=config.get('d_model', 384),
        n_heads=config.get('n_heads', 6),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1536),
        max_seq_len=config.get('max_seq_len', 1024),
        dropout=config.get('dropout', 0.1),
    )
