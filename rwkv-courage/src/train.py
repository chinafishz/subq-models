"""
RWKV-Courage Training Loop
============================
Main training script for RWKV-7 "Goose" language model.
MPS-compatible (Apple Silicon). No CUDA required.

Usage:
    python -m src.train --config configs/courage_25m.yaml
"""

import argparse
import os
import sys
import time
import math
from typing import Optional
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CourageLM, RWKV7Config
from src.dataset import MixedDataset
from tokenizers import Tokenizer


def get_device(config: dict) -> torch.device:
    """Auto-detect best available device: MPS > CUDA > CPU."""
    requested = config.get("training", {}).get("device", "auto")

    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif requested != "cpu":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")

    return torch.device("cpu")


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(
    model: CourageLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    output_dir: str,
):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint_{step:06d}.pt")

    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.config.__dict__,
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: CourageLM,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """Load model checkpoint. Returns the step number."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("step", 0)


def train(config_path: str, resume_from: Optional[str] = None):
    """Main training function."""
    # --- Load config ---
    cfg = load_config(config_path)
    model_cfg = RWKV7Config(cfg["model"])
    train_cfg = cfg["training"]

    device = get_device(cfg)
    dtype = torch.float32  # MPS safest
    print(f"Device: {device} | MPS available: {torch.backends.mps.is_available()}")

    # --- Load tokenizer ---
    tokenizer_path = cfg.get("tokenizer_path", "data/tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        print("  Run: python -m src.tokenizer_train --input <files> --output data/tokenizer.json")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}")

    # --- Create model ---
    model = CourageLM(model_cfg).to(device).to(dtype)
    n_params = model.count_parameters()
    print(f"Model: {n_params:,} parameters ({n_params / 1e6:.1f}M)")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.95)),
        weight_decay=train_cfg.get("weight_decay", 0.1),
    )

    # --- Data ---
    dataset = MixedDataset(
        tokenizer=tokenizer,
        tinystories_ratio=train_cfg["data_mix"]["tinystories"],
        courage_ratio=train_cfg["data_mix"]["courage_material"],
        seq_len=model_cfg.ctx_len,
    )

    courage_path = cfg.get("courage_material_path", "data/raw/courage_material.txt")
    data_stream = dataset.stream_tokens(courage_path)

    # --- Resume from checkpoint ---
    start_step = 0
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
    resume = resume_from or cfg.get("resume_from")
    if resume and os.path.exists(resume):
        start_step = load_checkpoint(resume, model, optimizer)
        print(f"Resumed from step {start_step}")

    # --- Training loop ---
    batch_size = train_cfg["batch_size"]
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    max_steps = train_cfg["max_steps"]
    warmup_steps = train_cfg["warmup_steps"]
    min_lr = train_cfg.get("min_lr", 3e-5)
    base_lr = train_cfg["learning_rate"]

    model.train()
    step = start_step
    total_loss = 0.0
    last_avg_loss = float("inf")  # tracked for checkpoint metadata
    best_loss = float("inf")
    t0 = time.time()

    print(f"\nTraining: batch_size={batch_size}, grad_accum={grad_accum}")
    print(f"  Effective batch = {batch_size * grad_accum}")
    print(f"  Max steps = {max_steps}\n")

    optimizer.zero_grad()

    while step < max_steps:
        step += 1

        # --- Learning rate schedule (cosine with warmup) ---
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- Get batch ---
        try:
            seq = next(data_stream)
        except StopIteration:
            data_stream = dataset.stream_tokens(courage_path)
            seq = next(data_stream)

        # seq shape: (seq_len+1,) -> split into input/target
        x = seq[:-1].view(1, -1).to(device)  # (1, seq_len)
        y = seq[1:].view(1, -1).to(device)   # (1, seq_len)

        # --- Forward ---
        logits = model(x)                     # (1, seq_len, vocab_size)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=tokenizer.token_to_id("[PAD]") or -100,
        )

        # NaN guard: abort training if loss becomes non-finite
        if not torch.isfinite(loss):
            print(f"ERROR: Non-finite loss ({loss.item()}) at step {step}. Aborting.")
            save_checkpoint(model, optimizer, step, float("nan"), checkpoint_dir)
            break

        loss = loss / grad_accum
        loss.backward()

        # --- Gradient accumulation ---
        if step % grad_accum == 0:
            # Gradient clipping
            if train_cfg.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum

        # --- Logging ---
        log_interval = train_cfg.get("log_interval", 50)
        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            last_avg_loss = avg_loss  # capture before reset, for checkpoint metadata
            elapsed = time.time() - t0
            tokens_per_sec = (log_interval * model_cfg.ctx_len * batch_size) / elapsed

            print(
                f"step {step:6d}/{max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"{tokens_per_sec:.0f} tok/s | "
                f"{elapsed:.1f}s"
            )

            total_loss = 0.0
            t0 = time.time()

            if avg_loss < best_loss:
                best_loss = avg_loss

        # --- Save checkpoint ---
        save_interval = train_cfg.get("save_interval", 1000)
        if step % save_interval == 0:
            save_checkpoint(model, optimizer, step, last_avg_loss,
                          checkpoint_dir)
            # Also save 'latest.pt' for easy resume
            latest_path = os.path.join(checkpoint_dir, "latest.pt")
            checkpoint = {
                "step": step,
                "loss": last_avg_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {k: v for k, v in model.config.__dict__.items()
                          if not k.startswith("_")},
            }
            torch.save(checkpoint, latest_path)

    # --- Final save ---
    final_path = os.path.join(checkpoint_dir, "final.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in model.config.__dict__.items() if not k.startswith("_")},
    }, final_path)
    print(f"\nTraining complete! Final model: {final_path}")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RWKV-Courage language model")
    parser.add_argument("--config", default="configs/courage_25m.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint path")
    args = parser.parse_args()

    train(config_path=args.config, resume_from=args.resume)
