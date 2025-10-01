#!/usr/bin/env python3
"""
LLM Visualization and Interpretability Script
This script provides comprehensive visualization tools for understanding LLM behavior:
- Attention pattern visualization (heatmaps)
- Token embedding projections (PCA, t-SNE)
- Layer-wise activation analysis
- Generation probability distributions
- Gradient flow analysis
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gpt import SmallGPT
from training.dataset import SimpleTokenizer
from utils.helpers import get_device


class LLMVisualizer:
    """Comprehensive visualization toolkit for LLM interpretability."""

    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.attention_cache = []
        self.activation_cache = {}

    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        hooks = []

        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activation_cache[name] = output[0].detach().cpu()
                else:
                    self.activation_cache[name] = output.detach().cpu()
            return hook

        # Register hooks for each transformer block
        for idx, block in enumerate(self.model.transformer_blocks):
            hooks.append(block.register_forward_hook(get_activation(f'block_{idx}')))

        return hooks

    def _remove_hooks(self, hooks):
        """Remove all registered hooks."""
        for hook in hooks:
            hook.remove()

    def visualize_attention_patterns(self, text, layer_idx=0, head_idx=0, save_path=None):
        """
        Visualize attention patterns for a specific layer and head.

        Args:
            text: Input text string
            layer_idx: Which transformer layer to visualize
            head_idx: Which attention head to visualize
            save_path: Optional path to save the figure
        """
        # Tokenize input
        tokens = self.tokenizer.encode(text)
        token_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

        # Forward pass to capture attention
        with torch.no_grad():
            x = self.model.embedding(token_ids)
            for idx, block in enumerate(self.model.transformer_blocks):
                x, attn_weights = block(x)
                if idx == layer_idx:
                    # attn_weights shape: (batch, n_heads, seq_len, seq_len)
                    attention_pattern = attn_weights[0, head_idx].cpu().numpy()
                    break

        # Decode tokens for labels
        token_strs = [self.tokenizer.decode([t]) for t in tokens]

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_pattern,
                   xticklabels=token_strs,
                   yticklabels=token_strs,
                   cmap='viridis',
                   ax=ax,
                   cbar_kws={'label': 'Attention Weight'})

        ax.set_title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}')
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_all_heads(self, text, layer_idx=0, save_path=None):
        """
        Visualize attention patterns for all heads in a layer.

        Args:
            text: Input text string
            layer_idx: Which transformer layer to visualize
            save_path: Optional path to save the figure
        """
        tokens = self.tokenizer.encode(text)
        token_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        token_strs = [self.tokenizer.decode([t]) for t in tokens]

        # Forward pass to get attention weights
        with torch.no_grad():
            x = self.model.embedding(token_ids)
            for idx, block in enumerate(self.model.transformer_blocks):
                x, attn_weights = block(x)
                if idx == layer_idx:
                    # attn_weights shape: (batch, n_heads, seq_len, seq_len)
                    all_heads = attn_weights[0].cpu().numpy()
                    break

        n_heads = all_heads.shape[0]
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_heads == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        for head_idx in range(n_heads):
            row = head_idx // n_cols
            col = head_idx % n_cols
            ax = axes[row, col]

            sns.heatmap(all_heads[head_idx],
                       xticklabels=token_strs,
                       yticklabels=token_strs,
                       cmap='viridis',
                       ax=ax,
                       cbar=False)
            ax.set_title(f'Head {head_idx}')
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Hide unused subplots
        for idx in range(n_heads, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        fig.suptitle(f'All Attention Heads - Layer {layer_idx}', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved multi-head attention visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_token_embeddings(self, texts, method='pca', save_path=None):
        """
        Visualize token embeddings in 2D using dimensionality reduction.

        Args:
            texts: List of text strings to visualize
            method: 'pca' or 'tsne'
            save_path: Optional path to save the figure
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
        except ImportError:
            print("Please install scikit-learn: pip install scikit-learn")
            return

        # Get embeddings for all tokens
        all_embeddings = []
        all_tokens = []

        for text in texts:
            tokens = self.tokenizer.encode(text)
            token_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

            with torch.no_grad():
                embeddings = self.model.embedding.token_embedding.embedding(token_ids)
                all_embeddings.append(embeddings[0].cpu().numpy())
                all_tokens.extend([self.tokenizer.decode([t]) for t in tokens])

        # Concatenate all embeddings
        embeddings_matrix = np.vstack(all_embeddings)

        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
            embeddings_2d = reducer.fit_transform(embeddings_matrix)
            title = 'Token Embeddings (PCA)'
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings_matrix)
            title = 'Token Embeddings (t-SNE)'
        else:
            raise ValueError(f"Unknown method: {method}")

        # Visualize
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           alpha=0.6, s=100, c=range(len(all_tokens)),
                           cmap='tab20')

        # Annotate points
        for i, token in enumerate(all_tokens):
            ax.annotate(token, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       fontsize=8, alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved embedding visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_generation_probabilities(self, context, top_k=10, save_path=None):
        """
        Visualize the probability distribution over next tokens.

        Args:
            context: Input context string
            top_k: Number of top predictions to show
            save_path: Optional path to save the figure
        """
        tokens = self.tokenizer.encode(context)
        token_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(token_ids)
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        # Decode tokens
        top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]

        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(top_k), top_probs)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_tokens)
        ax.set_xlabel('Probability')
        ax.set_title(f'Top {top_k} Next Token Predictions\nContext: "{context}"')
        ax.invert_yaxis()

        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, top_probs)):
            ax.text(prob, i, f' {prob:.4f}', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved probability visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_layer_activations(self, text, save_path=None):
        """
        Visualize activation statistics across all layers.

        Args:
            text: Input text string
            save_path: Optional path to save the figure
        """
        tokens = self.tokenizer.encode(text)
        token_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

        # Register hooks and perform forward pass
        hooks = self._register_hooks()

        with torch.no_grad():
            _ = self.model(token_ids)

        # Remove hooks
        self._remove_hooks(hooks)

        # Compute statistics for each layer
        layer_names = []
        mean_activations = []
        std_activations = []
        max_activations = []

        for name, activation in sorted(self.activation_cache.items()):
            layer_names.append(name)
            mean_activations.append(activation.mean().item())
            std_activations.append(activation.std().item())
            max_activations.append(activation.max().item())

        # Clear cache
        self.activation_cache = {}

        # Visualize
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        x_pos = range(len(layer_names))

        ax1.plot(x_pos, mean_activations, marker='o', linewidth=2)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(layer_names, rotation=45)
        ax1.set_ylabel('Mean Activation')
        ax1.set_title('Layer-wise Activation Statistics')
        ax1.grid(True, alpha=0.3)

        ax2.plot(x_pos, std_activations, marker='s', linewidth=2, color='orange')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(layer_names, rotation=45)
        ax2.set_ylabel('Std Deviation')
        ax2.grid(True, alpha=0.3)

        ax3.plot(x_pos, max_activations, marker='^', linewidth=2, color='green')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(layer_names, rotation=45)
        ax3.set_ylabel('Max Activation')
        ax3.set_xlabel('Layer')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved layer activation visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    def analyze_attention_entropy(self, text, save_path=None):
        """
        Compute and visualize attention entropy across layers and heads.
        Lower entropy = more focused attention, higher entropy = more diffuse.

        Args:
            text: Input text string
            save_path: Optional path to save the figure
        """
        tokens = self.tokenizer.encode(text)
        token_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

        # Collect attention weights from all layers
        all_entropies = []

        with torch.no_grad():
            x = self.model.embedding(token_ids)
            for idx, block in enumerate(self.model.transformer_blocks):
                x, attn_weights = block(x)
                # attn_weights shape: (batch, n_heads, seq_len, seq_len)

                # Compute entropy for each head
                attn = attn_weights[0].cpu().numpy()  # Remove batch dimension
                entropies = []

                for head_idx in range(attn.shape[0]):
                    head_attn = attn[head_idx]
                    # Compute entropy for each query position
                    eps = 1e-10
                    entropy = -np.sum(head_attn * np.log(head_attn + eps), axis=-1)
                    # Average entropy across positions
                    mean_entropy = entropy.mean()
                    entropies.append(mean_entropy)

                all_entropies.append(entropies)

        # Convert to numpy array for easier plotting
        entropy_matrix = np.array(all_entropies)

        # Visualize
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(entropy_matrix, cmap='coolwarm', aspect='auto')
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Layer')
        ax.set_title('Attention Entropy Across Layers and Heads\n(Lower = More Focused)')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Entropy', rotation=270, labelpad=20)

        # Add text annotations
        for i in range(entropy_matrix.shape[0]):
            for j in range(entropy_matrix.shape[1]):
                text = ax.text(j, i, f'{entropy_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention entropy visualization to {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    """Demo script showing various visualization capabilities."""

    # Load model and tokenizer
    print("Loading model and tokenizer...")

    checkpoint_path = 'checkpoints/best_model.pt'
    tokenizer_path = 'checkpoints/tokenizer.pkl'

    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train a model first using train_example.py")
        return

    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("Please train a model first using train_example.py")
        return

    # Load tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint.get('config', {
        'vocab_size': tokenizer.vocab_size,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 512,
        'max_seq_len': 256,
        'dropout': 0.1,
    })

    model = SmallGPT(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])

    device = get_device()
    print(f"Using device: {device}")

    # Create visualizer
    visualizer = LLMVisualizer(model, tokenizer, device)

    # Create output directory
    os.makedirs('visualizations', exist_ok=True)

    # Sample text for visualization
    sample_text = "Once upon a time, in a land far away"

    print("\n=== Generating Visualizations ===\n")

    # 1. Attention patterns for a specific head
    print("1. Visualizing attention patterns for Layer 0, Head 0...")
    visualizer.visualize_attention_patterns(
        sample_text,
        layer_idx=0,
        head_idx=0,
        save_path='visualizations/attention_pattern_L0_H0.png'
    )

    # 2. All attention heads in a layer
    print("2. Visualizing all attention heads in Layer 0...")
    visualizer.visualize_all_heads(
        sample_text,
        layer_idx=0,
        save_path='visualizations/all_heads_L0.png'
    )

    # 3. Generation probabilities
    print("3. Visualizing next token prediction probabilities...")
    visualizer.visualize_generation_probabilities(
        "Once upon a",
        top_k=15,
        save_path='visualizations/next_token_probs.png'
    )

    # 4. Layer activations
    print("4. Visualizing layer-wise activations...")
    visualizer.visualize_layer_activations(
        sample_text,
        save_path='visualizations/layer_activations.png'
    )

    # 5. Attention entropy
    print("5. Analyzing attention entropy...")
    visualizer.analyze_attention_entropy(
        sample_text,
        save_path='visualizations/attention_entropy.png'
    )

    # 6. Token embeddings (if sklearn is available)
    print("6. Visualizing token embeddings...")
    try:
        sample_texts = [
            "Once upon a time",
            "In the beginning",
            "Long ago there",
            "The story starts"
        ]
        visualizer.visualize_token_embeddings(
            sample_texts,
            method='pca',
            save_path='visualizations/token_embeddings_pca.png'
        )
    except Exception as e:
        print(f"   Skipping (requires scikit-learn): {e}")

    print("\n=== Visualization Complete ===")
    print("All visualizations saved to 'visualizations/' directory")


if __name__ == "__main__":
    main()