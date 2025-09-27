import torch
import random
import numpy as np
import os


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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