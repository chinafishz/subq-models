#!/usr/bin/env python3
"""
GSA Model Training — tokenizer + TinyStories + training loop.

Usage:
    # Full pipeline (train tokenizer + model)
    python train_gsa.py --steps 2000 --batch 4 --device mps

    # Resume from checkpoint
    python train_gsa.py --resume checkpoints/latest.pt --steps 5000
"""

import argparse
import os
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn.functional as F

# Paths
SPIKE_DIR = Path(__file__).parent
PROJECT_DIR = SPIKE_DIR.parent.parent
sys.path.insert(0, str(SPIKE_DIR))

from model_gsa import GSA_Config, GSALanguageModel

# ─── Tokenizer ────────────────────────────────────────────────────────────

def build_tokenizer(input_texts: list, vocab_size: int = 2000, output_path: str = None):
    """Train BPE tokenizer on input texts. Falls back to simple char-level if tokenizers unavailable."""
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.post_processor = processors.ByteLevel(trim_offsets=False)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        # Write texts to temp files
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for text in input_texts:
                f.write(text + '\n')
            tmp_path = f.name

        tok.train(files=[tmp_path], trainer=trainer)
        os.unlink(tmp_path)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            tok.save(output_path)

        print(f"BPE tokenizer: vocab={tok.get_vocab_size()} (saved to {output_path})")
        return tok, tok.get_vocab_size()

    except ImportError:
        print("tokenizers not installed — using simple char-level tokenizer")
        return SimpleTokenizer(vocab_size), vocab_size


class SimpleTokenizer:
    """Character-level fallback tokenizer."""
    def __init__(self, vocab_size: int = 2000):
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def encode(self, text: str):
        ids = [self.bos_id]
        for ch in text:
            tid = 4 + (ord(ch) % (self.vocab_size - 4))
            ids.append(min(tid, self.vocab_size - 1))
        return type('Encoded', (), {'ids': ids})()

    def decode(self, ids: list) -> str:
        chars = []
        for i in ids:
            if i <= 3:
                continue
            chars.append(chr(32 + (i - 4) % 95))
        return ''.join(chars)

    def token_to_id(self, token: str):
        mapping = {'[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3}
        return mapping.get(token, None)

    def get_vocab_size(self):
        return self.vocab_size

    def save(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump({'vocab_size': self.vocab_size}, f)

    @staticmethod
    def from_file(path: str):
        import json
        with open(path) as f:
            d = json.load(f)
        return SimpleTokenizer(d['vocab_size'])


# ─── Data ─────────────────────────────────────────────────────────────────

def get_tinystories_data(max_texts: int = 5000):
    """Download TinyStories from HF or use bundled samples."""
    texts = []
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True,
                          trust_remote_code=False)
        for i, item in enumerate(ds):
            if i >= max_texts:
                break
            texts.append(item["text"])
        print(f"Loaded {len(texts)} TinyStories from HuggingFace")
        return texts
    except Exception as e:
        print(f"HuggingFace unavailable ({e}), using built-in samples")
        return _get_sample_texts(max_texts)


def _get_sample_texts(n: int):
    """Fallback: simple English samples for training."""
    samples = [
        "Once upon a time, a brave little cat went to explore the forest.",
        "I believe that courage is the first step to any great adventure.",
        "The sun set behind the mountains as the travelers made camp.",
        "She opened the old book and found a secret message inside.",
        "The robot learned to paint beautiful pictures of the stars.",
        "A kind dragon helped the villagers rebuild their homes after the storm.",
        "Deep in the ocean, a curious fish discovered a glowing crystal.",
        "The young wizard practiced her spells every morning at dawn.",
        "They walked together through the garden, talking about their dreams.",
        "The little bird sang a song that made everyone in the forest smile.",
        "He fixed the broken machine using only a screwdriver and some tape.",
        "The moon was full and bright, casting silver light on the water.",
        "She decided to try again, even though she had failed many times.",
        "A mysterious package arrived at the doorstep on a rainy Tuesday.",
        "The brave knight faced the dragon, not with a sword, but with kindness.",
        "They built a treehouse high up in the old oak tree.",
        "The scientist made a discovery that would change everything.",
        "Every night, the lighthouse keeper would light the great lamp.",
        "The fox taught the rabbit how to find the sweetest berries.",
        "I want to be a doctor so I can help people feel better.",
        "A shooting star streaked across the sky and they all made a wish.",
        "The old clock tower had been silent for a hundred years.",
        "She picked up the paintbrush and began to create a masterpiece.",
        "The wind carried the seeds far across the meadow to new ground.",
        "He wrote a letter to his future self and buried it under the apple tree.",
    ]
    # Repeat to reach n
    result = []
    while len(result) < n:
        result.extend(samples)
    return result[:n]


# ─── Training Loop ────────────────────────────────────────────────────────

def train(config: dict):
    """Main training function."""
    cfg_model = config["model"]
    cfg_train = config["training"]

    # Device
    device_str = cfg_train.get("device", "cpu")
    if device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device} | MPS: {torch.backends.mps.is_available()}")

    # ── Data ──
    texts = get_tinystories_data(cfg_train.get("max_texts", 2000))
    tok_path = cfg_train.get("tokenizer_path", "data/gsa_tokenizer.json")
    tokenizer, vocab_size = build_tokenizer(texts, cfg_model["vocab_size"], tok_path)

    cfg_model["vocab_size"] = vocab_size
    gsa_cfg = GSA_Config(cfg_model)

    # Pre-tokenize all texts
    tokenized = []
    for text in texts:
        encoded = tokenizer.encode(text)
        tokenized.append(torch.tensor(encoded.ids, dtype=torch.long))

    print(f"Tokenized {len(tokenized)} texts, vocab={vocab_size}")

    # ── Model ──
    model = GSALanguageModel(gsa_cfg).to(device)
    n_params = model.count_parameters()
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")

    # ── Optimizer ──
    lr = cfg_train.get("learning_rate", 3e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(cfg_train.get("beta1", 0.9), cfg_train.get("beta2", 0.95)),
        weight_decay=cfg_train.get("weight_decay", 0.1),
    )

    # ── Resume ──
    start_step = 0
    checkpoint_dir = cfg_train.get("checkpoint_dir", "checkpoints_gsa")
    resume_path = cfg_train.get("resume_from", None)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    # ── Training params ──
    batch_size = cfg_train.get("batch_size", 4)
    grad_accum = cfg_train.get("gradient_accumulation_steps", 8)
    max_steps = cfg_train.get("max_steps", 2000)
    warmup_steps = cfg_train.get("warmup_steps", 100)
    min_lr = cfg_train.get("min_lr", 3e-5)
    ctx_len = min(cfg_model.get("ctx_len", 256), 256)  # cap for speed
    grad_clip = cfg_train.get("grad_clip", 1.0)

    model.train()
    opt_step = start_step
    optimizer.zero_grad()
    total_loss = 0.0
    best_loss = float("inf")
    t0 = time.time()

    print(f"\nTraining: bs={batch_size}, accum={grad_accum}, "
          f"eff_bs={batch_size*grad_accum}")
    print(f"  ctx={ctx_len}, steps={start_step}→{max_steps}, lr={lr}\n")

    # Main loop
    for step in range(start_step, max_steps):
        # Build batch from random text samples
        x_batch = torch.zeros(batch_size, ctx_len, dtype=torch.long, device=device)
        y_batch = torch.zeros(batch_size, ctx_len, dtype=torch.long, device=device)

        for b in range(batch_size):
            # Pick random text, take random slice
            idx_text = torch.randint(0, len(tokenized), (1,)).item()
            tokens = tokenized[idx_text]
            if len(tokens) <= ctx_len + 1:
                # Too short: pad
                x_batch[b, :len(tokens)-1] = tokens[:-1]
                y_batch[b, :len(tokens)-1] = tokens[1:]
                y_batch[b, len(tokens)-1:] = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0
            else:
                start = torch.randint(0, len(tokens) - ctx_len - 1, (1,)).item()
                x_batch[b] = tokens[start:start + ctx_len]
                y_batch[b] = tokens[start + 1:start + ctx_len + 1]

        # Forward
        logits = model(x_batch)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            y_batch.view(-1),
            ignore_index=-100,
        )

        if not torch.isfinite(loss):
            print(f"ERROR: NaN loss at step {step + 1}. Aborting.")
            break

        loss = loss / grad_accum
        loss.backward()

        # Optimizer step
        if (step + 1) % grad_accum == 0:
            opt_step += 1

            # LR schedule (cosine + warmup)
            if opt_step < warmup_steps:
                current_lr = lr * opt_step / max(1, warmup_steps)
            else:
                progress = (opt_step - warmup_steps) / max(1, max_steps // grad_accum - warmup_steps)
                current_lr = min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum

        # Logging
        log_interval = cfg_train.get("log_interval", 50)
        if (step + 1) % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - t0
            print(f"step {step+1:5d}/{max_steps} | loss {avg_loss:.4f} | "
                  f"lr {current_lr:.2e} | {elapsed:.1f}s")
            total_loss = 0.0
            t0 = time.time()

            if avg_loss < best_loss:
                best_loss = avg_loss

    # ── Save final ──
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_path = os.path.join(checkpoint_dir, "final.pt")
    torch.save({
        "step": step + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {k: v for k, v in gsa_cfg.__dict__.items() if not k.startswith("_")},
    }, final_path)
    print(f"\nModel saved: {final_path}")
    print(f"Best loss: {best_loss:.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GSA Language Model")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--tokenizer", default="data/gsa_tokenizer.json")
    args = parser.parse_args()

    config = {
        "model": {
            "n_embd": 384,
            "n_layer": 8,
            "vocab_size": 2000,
            "ctx_len": args.ctx,
            "n_head": 6,
            "head_dim": 64,
            "global_k": 64,
            "window": 32,
        },
        "training": {
            "batch_size": args.batch,
            "gradient_accumulation_steps": 4,
            "learning_rate": 3e-4,
            "min_lr": 3e-5,
            "warmup_steps": 50,
            "max_steps": args.steps,
            "max_texts": max(500, args.steps * 2),
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "log_interval": 20,
            "save_interval": 500,
            "device": args.device,
            "tokenizer_path": args.tokenizer,
            "checkpoint_dir": "checkpoints_gsa",
            "resume_from": args.resume,
        },
    }

    train(config)


if __name__ == "__main__":
    main()
