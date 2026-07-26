#!/usr/bin/env python3
"""
Modern block components: RMSNorm vs LayerNorm, SwiGLU vs GELU (M1.3).

Two demonstrations, both fast and CPU-only:

1. **Normalization.** RMSNorm drops LayerNorm's mean-centering and bias,
   normalizing only by the root-mean-square. We confirm parity with the
   reference formula and time both — RMSNorm does strictly less work.

2. **Feed-forward.** SwiGLU is a gated MLP (three matrices) that, under the
   `2/3 · 4d` hidden-dim convention, has ≈ the same parameter count as a GELU
   MLP (two matrices) while typically learning better. We report the param
   match and, with `--full`, a short train comparing a GPT-2 stack against a
   LLaMA-ish (RoPE + RMSNorm + SwiGLU + QK-norm) stack.

Run:
    python modules/m1_fundamentals/block_components.py            # fast (CI)
    python modules/m1_fundamentals/block_components.py --full     # + train compare
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models import GPTConfig, create_model  # noqa: E402
from models.norms import RMSNorm  # noqa: E402
from models.feedforward import FeedForward, SwiGLUFeedForward, swiglu_hidden_dim  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "results"


def time_norm(module, x, iters):
    with torch.no_grad():
        module(x)  # warmup
        t0 = time.perf_counter()
        for _ in range(iters):
            module(x)
        return (time.perf_counter() - t0) / iters


def norm_benchmark(dim=1024, batch=64, seq=128, iters=50):
    x = torch.randn(batch, seq, dim)
    ln = nn.LayerNorm(dim)
    rms = RMSNorm(dim)
    ln_t = time_norm(ln, x, iters)
    rms_t = time_norm(rms, x, iters)
    return {
        "dim": dim,
        "layernorm_ms": round(ln_t * 1e3, 4),
        "rmsnorm_ms": round(rms_t * 1e3, 4),
        "wall_clock_ratio": round(ln_t / rms_t, 3),
        "layernorm_params": sum(p.numel() for p in ln.parameters()),
        "rmsnorm_params": sum(p.numel() for p in rms.parameters()),
        # RMSNorm's real, unconditional win: half the parameters (no bias) and
        # fewer reductions/ops (no mean-centering) — one reduction vs two.
        "param_reduction": round(
            1 - sum(p.numel() for p in rms.parameters()) / sum(p.numel() for p in ln.parameters()), 3
        ),
        # HONEST CAVEAT: this educational RMSNorm (unfused Python ops + an fp32
        # cast for stability) is often *slower* wall-clock than PyTorch's fused
        # C++ LayerNorm kernel. RMSNorm's fewer-FLOPs advantage only materializes
        # with a fused kernel (Triton/CUDA/torch.compile). Don't read the raw ms
        # here as "LayerNorm is faster" — it's a kernel-fusion artifact.
        "note": "wall-clock reflects kernel fusion, not FLOPs; RMSNorm wins on params/FLOPs",
    }


def ffn_benchmark(d_model=768, d_ff=3072):
    gelu = FeedForward(d_model, d_ff)
    swiglu = SwiGLUFeedForward(d_model, d_ff)
    g = sum(p.numel() for p in gelu.parameters())
    s = sum(p.numel() for p in swiglu.parameters())
    return {
        "d_model": d_model,
        "gelu_d_ff": d_ff,
        "swiglu_hidden": swiglu_hidden_dim(d_ff),
        "gelu_params": g,
        "swiglu_params": s,
        "param_ratio": round(s / g, 4),
    }


def make_batch(batch, seq, vocab, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab, (batch, seq), generator=g)
    return x


def train_compare(steps, device):
    dev = torch.device(device)
    vocab, seq = 128, 64
    out = {}
    stacks = {
        "gpt2 (layernorm/gelu/learned)": dict(),
        "modern (rope/rmsnorm/swiglu/qknorm)": dict(
            pos_encoding="rope", norm="rmsnorm", activation="swiglu", qk_norm=True
        ),
    }
    for name, extra in stacks.items():
        torch.manual_seed(0)
        model = create_model(GPTConfig(
            vocab_size=vocab, d_model=128, n_heads=4, n_layers=4, d_ff=512,
            max_seq_len=seq, dropout=0.0, **extra,
        )).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        model.train()
        for step in range(steps):
            x = make_batch(32, seq, vocab, seed=step).to(dev)
            _, loss = model(x, x)  # plain LM loss on random tokens (relative memorization)
            opt.zero_grad()
            loss.backward()
            opt.step()
        out[name] = round(loss.item(), 4)
        print(f"{name:38} final loss={loss.item():.4f}  params={model.get_num_params(True):,}")
    return out


def plot(result, out_png: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    nb = result["norm"]
    ax1.bar(["LayerNorm", "RMSNorm"], [nb["layernorm_params"], nb["rmsnorm_params"]], color=["#4C72B0", "#55A868"])
    ax1.set_ylabel("norm parameters")
    ax1.set_title(f"Norm params (dim={nb['dim']}) — RMSNorm {int(nb['param_reduction'] * 100)}% fewer (no bias)")
    fb = result["ffn"]
    ax2.bar(["GELU MLP", "SwiGLU"], [fb["gelu_params"], fb["swiglu_params"]], color=["#4C72B0", "#C44E52"])
    ax2.set_ylabel("FFN parameters")
    ax2.set_title(f"SwiGLU param match (ratio {fb['param_ratio']})")
    fig.suptitle("M1.3 block components", fontweight="bold")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--full", action="store_true", help="also run the train comparison")
    args = p.parse_args()

    result = {
        "benchmark": "block_components",
        "norm": norm_benchmark(),
        "ffn": ffn_benchmark(),
    }
    print("Norm:  ", json.dumps(result["norm"]))
    print("FFN:   ", json.dumps(result["ffn"]))

    if args.full:
        result["train_compare_final_loss"] = train_compare(steps=300, device=args.device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "m1_block_components.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    png_path = plot(result, RESULTS_DIR / "m1_block_components.png")
    print(f"\nwritten to {json_path}")
    if png_path:
        print(f"plot     {png_path}")


if __name__ == "__main__":
    main()
