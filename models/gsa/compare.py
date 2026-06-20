#!/usr/bin/env python3
"""
Spike 001: GSA vs RWKV-7 training comparison.

Trains both models on identical synthetic data for 100 steps.
Compares: loss curve, training speed, parameter count.

Usage:
    python compare.py [--steps 100] [--ctx 128] [--batch 2]
"""

import argparse
import os
import sys
import time
import types
import importlib.util
from pathlib import Path

# Path setup
SPIKE_DIR = Path(__file__).parent
PROJECT_DIR = SPIKE_DIR.parent.parent
SRC_DIR = str(PROJECT_DIR / "rwkv-courage" / "src")

import torch
import torch.nn.functional as F

# Import RWKV-7 model (handles relative import from src/ package)
spec_wkv = importlib.util.spec_from_file_location(
    "wkv7_operator", os.path.join(SRC_DIR, "wkv7_operator.py"))
wkv7 = importlib.util.module_from_spec(spec_wkv)
sys.modules["wkv7_operator"] = wkv7
spec_wkv.loader.exec_module(wkv7)

# Create fake package for src/ to satisfy relative imports
src_pkg = types.ModuleType("src")
src_pkg.wkv7_operator = wkv7
sys.modules["src"] = src_pkg
sys.modules["src.wkv7_operator"] = wkv7

spec_model = importlib.util.spec_from_file_location(
    "src.model", os.path.join(SRC_DIR, "model.py"))
model_mod = importlib.util.module_from_spec(spec_model)
sys.modules["src.model"] = model_mod
spec_model.loader.exec_module(model_mod)

CourageLM = model_mod.CourageLM
RWKV7Config = model_mod.RWKV7Config

from model_gsa import GSA_Config, GSALanguageModel


def build_configs(vocab_size=8000, ctx_len=128):
    """Build matching configs for fair comparison."""
    base = {
        "n_embd": 384,
        "n_layer": 8,
        "vocab_size": vocab_size,
        "ctx_len": ctx_len,
        "n_head": 6,
        "head_dim": 64,
        "global_k": 64,
        "window": 32,
    }

    rwkv_cfg = RWKV7Config({
        "n_embd": base["n_embd"],
        "n_layer": base["n_layer"],
        "vocab_size": base["vocab_size"],
        "ctx_len": base["ctx_len"],
        "head_size_a": base["head_dim"],
        "D_DECAY_LORA": 32,
        "D_AAA_LORA": 32,
        "D_MV_LORA": 16,
        "D_GATE_LORA": 64,
    })

    gsa_cfg = GSA_Config(base)
    return rwkv_cfg, gsa_cfg


def train_steps(model, optimizer, steps, ctx_len, batch_size, device, dtype, label=""):
    """Run training steps, return list of (step, loss, elapsed)."""
    model.train()
    records = []
    t0 = time.time()

    # Pre-generate synthetic data (same for both models)
    torch.manual_seed(42)  # reproducible
    # Handle different config access patterns
    if hasattr(model, 'cfg'):
        vocab_size = model.cfg.vocab_size
    else:
        vocab_size = model.config.vocab_size
    data = torch.randint(0, vocab_size, (steps * batch_size, ctx_len + 1))

    for step in range(steps):
        # Batch
        x_batch = torch.zeros(batch_size, ctx_len, dtype=torch.long, device=device)
        y_batch = torch.zeros(batch_size, ctx_len, dtype=torch.long, device=device)
        for b in range(batch_size):
            idx = step * batch_size + b
            x_batch[b] = data[idx, :ctx_len]
            y_batch[b] = data[idx, 1:ctx_len+1]

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            y_batch.view(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        records.append((step + 1, loss.item(), time.time() - t0))

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")  # use cpu for spike safety
    args = parser.parse_args()

    # Detect device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dtype = torch.float32
    vocab_size = 8000
    ctx_len = args.ctx
    batch_size = args.batch
    steps = args.steps

    print(f"=== Spike 001: GSA vs RWKV-7 ===")
    print(f"Device: {device} | ctx={ctx_len} | batch={batch_size} | steps={steps}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()

    # Build models
    rwkv_cfg, gsa_cfg = build_configs(vocab_size, ctx_len)

    rwkv_model = CourageLM(rwkv_cfg).to(device).to(dtype)
    gsa_model = GSALanguageModel(gsa_cfg).to(device).to(dtype)

    print(f"RWKV-7 params: {rwkv_model.count_parameters():,} ({rwkv_model.count_parameters()/1e6:.1f}M)")
    print(f"GSA params:   {gsa_model.count_parameters():,} ({gsa_model.count_parameters()/1e6:.1f}M)")
    print()

    # Identical optimizers
    rwkv_opt = torch.optim.AdamW(rwkv_model.parameters(), lr=3e-4)
    gsa_opt = torch.optim.AdamW(gsa_model.parameters(), lr=3e-4)

    # --- Train RWKV-7 ---
    print("Training RWKV-7...")
    rwkv_records = train_steps(rwkv_model, rwkv_opt, steps, ctx_len, batch_size,
                               device, dtype, "RWKV-7")

    # --- Train GSA ---
    print("Training GSA...")
    gsa_records = train_steps(gsa_model, gsa_opt, steps, ctx_len, batch_size,
                              device, dtype, "GSA")

    # --- Report ---
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    rwkv_start = rwkv_records[0][1]
    rwkv_end = rwkv_records[-1][1]
    rwkv_time = rwkv_records[-1][2]

    gsa_start = gsa_records[0][1]
    gsa_end = gsa_records[-1][1]
    gsa_time = gsa_records[-1][2]

    print(f"{'':20} {'RWKV-7':>12} {'GSA':>12} {'Ratio':>10}")
    print(f"{'─'*20} {'─'*12} {'─'*12} {'─'*10}")
    print(f"{'Params':20} {rwkv_model.count_parameters():>12,} {gsa_model.count_parameters():>12,} {gsa_model.count_parameters()/rwkv_model.count_parameters():>9.2f}x")
    print(f"{'Step 1 loss':20} {rwkv_start:>12.4f} {gsa_start:>12.4f}")
    print(f"{'Step N loss':20} {rwkv_end:>12.4f} {gsa_end:>12.4f}")
    print(f"{'Loss reduction':20} {rwkv_start - rwkv_end:>11.4f} {gsa_start - gsa_end:>11.4f}")
    print(f"{'Total time':20} {rwkv_time:>11.1f}s {gsa_time:>11.1f}s {gsa_time/rwkv_time:>9.2f}x")
    print(f"{'Time/step':20} {rwkv_time/steps*1000:>9.0f}ms {gsa_time/steps*1000:>9.0f}ms")

    # Per-step loss trace (first 10 + last)
    print()
    print(f"{'Step':>6} {'RWKV-7 loss':>12} {'GSA loss':>12}")
    for i in range(min(10, len(rwkv_records))):
        s_r, l_r, _ = rwkv_records[i]
        s_g, l_g, _ = gsa_records[i]
        print(f"{s_r:>6} {l_r:>12.4f} {l_g:>12.4f}")

    if steps > 10:
        print(f"  ...")
    if steps > 20:
        # Last 5 steps
        for i in range(max(0, steps - 5), steps):
            s_r, l_r, _ = rwkv_records[i]
            s_g, l_g, _ = gsa_records[i]
            print(f"{s_r:>6} {l_r:>12.4f} {l_g:>12.4f}")

    print()
    print("Done. See README.md for verdict.")


if __name__ == "__main__":
    main()
