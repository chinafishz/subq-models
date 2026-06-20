#!/usr/bin/env python3
"""
GSA Chat Demo — 加载训练好的 GSA 模型进行对话生成。

用法:
    # 模式 1: 加载 checkpoint
    python demo_chat.py --checkpoint checkpoints/final.pt --prompt "I believe"

    # 模式 2: 未训练模型测试管道（输出随机）
    python demo_chat.py --untrained --prompt "Hello"

    # 模式 3: 交互模式
    python demo_chat.py --checkpoint checkpoints/final.pt --interactive

依赖:
    pip install tokenizers  (用于 BPE tokenizer，如未安装则用简单空格分词)
"""

import argparse
import os
import sys
import types
import importlib.util
from pathlib import Path

# Path setup
SPIKE_DIR = Path(__file__).parent
PROJECT_DIR = SPIKE_DIR.parent.parent
SRC_DIR = str(PROJECT_DIR / "rwkv-courage" / "src")

import torch

# Import GSA model
from model_gsa import GSA_Config, GSALanguageModel


def get_tokenizer(tokenizer_path: str = None):
    """加载或创建简单的 tokenizer。"""
    if tokenizer_path and os.path.exists(tokenizer_path):
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(tokenizer_path)
            return tok, "BPE"
        except ImportError:
            print("[WARN] tokenizers 库未安装，使用简单空格分词器")
        except Exception as e:
            print(f"[WARN] 无法加载 tokenizer: {e}，使用简单空格分词器")

    # Fallback: 简单字符级+空格分词器
    return SimpleTokenizer(), "simple"


class SimpleTokenizer:
    """简单字符级分词器，仅用于 demo。
    输出 token ID 在 [0, vocab_size) 范围内。
    """

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def encode(self, text: str) -> list:
        ids = [self.bos_id]
        for ch in text:
            # 映射到 [4, vocab_size) 范围内
            tid = 4 + (ord(ch) % (self.vocab_size - 4))
            ids.append(min(tid, self.vocab_size - 1))
        return ids

    def decode(self, ids: list) -> str:
        chars = []
        for i in ids:
            if i <= 3 or i >= self.vocab_size:
                continue
            # 反向映射（近似）
            c = chr(32 + (i - 4) % 95)
            chars.append(c)
        return ''.join(chars)


def load_model(checkpoint_path: str, device: torch.device):
    """从 checkpoint 加载模型。"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from checkpoint
    if "config" in ckpt:
        cfg_dict = ckpt["config"]
    else:
        # Default config for spike models
        cfg_dict = {
            "n_embd": 384, "n_layer": 8, "vocab_size": 8000,
            "ctx_len": 4096, "n_head": 6, "head_dim": 64,
            "global_k": 64, "window": 32,
        }

    cfg = GSA_Config(cfg_dict)
    model = GSALanguageModel(cfg).to(device)

    # Load weights
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        step = ckpt.get("step", "?")
        loss = ckpt.get("loss", "?")
    else:
        # Direct state dict
        model.load_state_dict(ckpt)
        step, loss = "?", "?"

    model.eval()
    return model, step, loss


def create_untrained_model(device: torch.device):
    """创建未训练的 GSA 模型（用于管道测试）。"""
    cfg = GSA_Config({
        "n_embd": 384, "n_layer": 8, "vocab_size": 8000,
        "ctx_len": 4096, "n_head": 6, "head_dim": 64,
        "global_k": 64, "window": 32,
    })
    model = GSALanguageModel(cfg).to(device)
    model.eval()
    return model


def generate_response(model, tokenizer, prompt: str, device,
                      max_tokens: int = 100, temperature: float = 0.8):
    """生成对 prompt 的续写。"""
    if hasattr(tokenizer, 'encode'):
        if isinstance(tokenizer, SimpleTokenizer):
            ids = tokenizer.encode(prompt)
        else:
            encoded = tokenizer.encode(prompt)
            ids = encoded.ids
            # BPE tokenizer: prepend BOS if available
            try:
                bos_id = tokenizer.token_to_id("[BOS]")
                if bos_id:
                    ids = [bos_id] + ids
            except Exception:
                pass
    else:
        ids = [ord(c) for c in prompt]

    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    output_tensor = model.generate(
        input_tensor,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    output_ids = output_tensor[0].tolist()[len(ids):]  # 只取生成部分

    if hasattr(tokenizer, 'decode'):
        text = tokenizer.decode(output_ids)
    else:
        text = ''.join(chr(i) if 32 <= i < 127 else ' ' for i in output_ids)

    return text


def main():
    parser = argparse.ArgumentParser(description="GSA Chat Demo")
    parser.add_argument("--checkpoint", help="模型 checkpoint 路径")
    parser.add_argument("--untrained", action="store_true",
                        help="使用未训练模型（管道测试）")
    parser.add_argument("--tokenizer", help="BPE tokenizer.json 路径")
    parser.add_argument("--prompt", default="I believe",
                        help="起始 prompt")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="采样温度")
    parser.add_argument("--interactive", action="store_true",
                        help="交互对话模式")
    parser.add_argument("--device", default="cpu",
                        help="设备: cpu, mps, cuda")
    args = parser.parse_args()

    # Device
    if args.device == "auto" or args.device == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"GSA Chat Demo")
    print("=" * 50)

    # Load model
    if args.checkpoint:
        model, step, loss = load_model(args.checkpoint, device)
        print(f"Loaded checkpoint (step {step}, loss {loss})")
    elif args.untrained:
        model = create_untrained_model(device)
        print("Using UNTRAINED model (random weights — output will be random)")
    else:
        print("ERROR: 需要 --checkpoint 或 --untrained")
        sys.exit(1)

    n_params = model.count_parameters()
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")
    print(f"Context window: {model.cfg.ctx_len} tokens")
    print()

    # Load tokenizer
    tokenizer, tok_type = get_tokenizer(args.tokenizer)
    if hasattr(tokenizer, 'get_vocab_size'):
        vocab = tokenizer.get_vocab_size()
    else:
        vocab = getattr(tokenizer, 'vocab_size', '?')
    print(f"Tokenizer: {tok_type} (vocab={vocab})")
    print()

    if args.interactive:
        # Interactive mode
        print("交互模式 — 输入 prompt 生成回复，输入 /quit 退出")
        print("-" * 50)
        while True:
            try:
                prompt = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if prompt.lower() in ('/quit', '/exit', '/q'):
                print("再见！")
                break
            if not prompt:
                continue

            print("GSA: ", end="", flush=True)
            response = generate_response(
                model, tokenizer, prompt, device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(response)
    else:
        # Single prompt mode
        print(f"Prompt: {args.prompt}")
        print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
        print("-" * 50)
        print("GSA: ", end="", flush=True)
        response = generate_response(
            model, tokenizer, args.prompt, device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(response)


if __name__ == "__main__":
    main()
