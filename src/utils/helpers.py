import torch
import random
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_params(model, non_embedding=False):
    """
    Total parameter count, optionally excluding embedding tables.

    non_embedding=True excludes nn.Embedding weights (token + position
    tables), which is the convention for comparing transformer sizes
    across vocab sizes (GPT-2's "124M" counts embeddings; scaling-law
    N usually doesn't).
    """
    n_params = sum(p.numel() for p in model.parameters())
    if non_embedding:
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                n_params -= module.weight.numel()
    return n_params


def estimate_flops(model, seq_len, non_embedding_params=None):
    """
    Estimate forward-pass FLOPs per token (PaLM appendix B approximation).

    forward FLOPs/token ≈ 2*N + 2*n_layers*seq_len*d_model
      - 2*N: every non-embedding parameter does one multiply-accumulate
      - the second term is the attention-matrix work, which grows with
        sequence length (the O(n^2) cost that KV caching / flash address)

    Training costs ~3x the forward pass (backward ≈ 2x forward), giving
    the familiar C ≈ 6*N*D scaling-law estimate.

    Returns a dict with per-token and per-sequence forward FLOPs.
    """
    n = non_embedding_params if non_embedding_params is not None else count_params(model, non_embedding=True)
    n_layers = getattr(model, 'n_layers', None)
    d_model = getattr(model, 'd_model', None)
    if n_layers is None or d_model is None:
        raise ValueError("model must expose n_layers and d_model attributes")

    flops_per_token = 2 * n + 2 * n_layers * seq_len * d_model
    return {
        'params_non_embedding': n,
        'flops_per_token_fwd': flops_per_token,
        'flops_per_seq_fwd': flops_per_token * seq_len,
        'flops_per_token_train': 3 * flops_per_token,
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def print_model_info(model):
    print(f"Model has {count_parameters(model):,} trainable parameters")
    print(f"Model size: {count_parameters(model) * 4 / 1024 / 1024:.2f} MB (float32)")


def save_model_config(model, path):
    config = {
        'vocab_size': model.vocab_size,
        'd_model': model.d_model,
        'n_heads': model.n_heads,
        'n_layers': model.n_layers,
        'max_seq_len': model.max_seq_len,
    }

    import json
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_model_config(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)


def create_learning_rate_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)