"""
RWKV-7 "Goose" Language Model (Pure PyTorch, MPS-compatible)
=============================================================
Adapted from BlinkDL/RWKV-LM RWKV-v7/rwkv_v7_demo.py

Key differences from official code:
1. No CUDA kernel dependency — uses pure PyTorch wkv7_operator
2. No torch.jit.script — plain nn.Module for MPS compatibility
3. Simplified value residual mechanism
4. FP32 by default (MPS BF16 support is limited)

License: Apache 2.0 (inherited from RWKV-LM)
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .wkv7_operator import wkv7_forward


class RWKV7Config:
    """Configuration container matching RWKV-7 args namespace."""
    def __init__(self, config_dict: dict):
        self.n_embd = config_dict["n_embd"]
        self.n_layer = config_dict["n_layer"]
        self.vocab_size = config_dict["vocab_size"]
        self.ctx_len = config_dict.get("ctx_len", 4096)
        self.head_size_a = config_dict.get("head_size_a", 64)
        self.D_DECAY_LORA = config_dict.get("D_DECAY_LORA", 32)
        self.D_AAA_LORA = config_dict.get("D_AAA_LORA", 32)
        self.D_MV_LORA = config_dict.get("D_MV_LORA", 16)
        self.D_GATE_LORA = config_dict.get("D_GATE_LORA", 64)

        # Derived
        self.dim_att = self.n_embd
        self.dim_ffn = int(self.n_embd * 3.5)  # slightly smaller than 4x for 25M budget
        self.head_size = self.head_size_a
        self.n_head = self.dim_att // self.head_size
        assert self.dim_att % self.n_head == 0, (
            f"dim_att ({self.dim_att}) must be divisible by n_head ({self.n_head})"
        )


class RWKV_Tmix_x070(nn.Module):
    """RWKV-7 Time Mixing block (the "attention" replacement)."""

    def __init__(self, args: RWKV7Config, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size
        self.n_head = args.n_head
        C = args.n_embd
        H = self.n_head
        N = self.head_size

        # Time-shift parameters (learned mixing of current and previous token)
        self.x_r = nn.Parameter(torch.ones(1, 1, C))
        self.x_w = nn.Parameter(torch.ones(1, 1, C))
        self.x_k = nn.Parameter(torch.ones(1, 1, C))
        self.x_v = nn.Parameter(torch.ones(1, 1, C))
        self.x_a = nn.Parameter(torch.ones(1, 1, C))
        self.x_g = nn.Parameter(torch.ones(1, 1, C))

        # Decay path (w): low-rank projection
        D_DECAY = args.D_DECAY_LORA
        self.w0 = nn.Parameter(torch.zeros(1, 1, C))
        self.w1 = nn.Parameter(torch.zeros(C, D_DECAY))
        self.w2 = nn.Parameter(torch.zeros(D_DECAY, C))

        # In-context learning rate path (a)
        D_AAA = args.D_AAA_LORA
        self.a0 = nn.Parameter(torch.zeros(1, 1, C))
        self.a1 = nn.Parameter(torch.zeros(C, D_AAA))
        self.a2 = nn.Parameter(torch.zeros(D_AAA, C))

        # Value residual path (v)
        D_MV = args.D_MV_LORA
        self.v0 = nn.Parameter(torch.zeros(1, 1, C)) if layer_id > 0 else None
        self.v1 = nn.Parameter(torch.zeros(C, D_MV)) if layer_id > 0 else None
        self.v2 = nn.Parameter(torch.zeros(D_MV, C)) if layer_id > 0 else None

        # Gate path (g)
        D_GATE = args.D_GATE_LORA
        self.g1 = nn.Parameter(torch.zeros(C, D_GATE))
        self.g2 = nn.Parameter(torch.zeros(D_GATE, C))

        # Key normalization parameters
        self.k_k = nn.Parameter(torch.ones(1, 1, C))
        self.k_a = nn.Parameter(torch.ones(1, 1, C))
        self.r_k = nn.Parameter(torch.zeros(H, N))

        # Main projections
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """RWKV-style initialization: most linear layers zero-init."""
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.w1)
        nn.init.zeros_(self.w2)
        nn.init.zeros_(self.a1)
        nn.init.zeros_(self.a2)
        if self.v1 is not None:
            nn.init.zeros_(self.v1)
            nn.init.zeros_(self.v2)
        nn.init.zeros_(self.g1)
        nn.init.zeros_(self.g2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, N = self.n_head, self.head_size

        # Time-shift: mix current and previous token
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        # 1. Receptance (query): what to read
        r = self.receptance(xr)

        # 2. Decay (w): how fast to forget — soft-clamped to (-inf, -0.5)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5

        # 3. Key and Value
        k = self.key(xk)
        v = self.value(xv)

        # 4. Value residual (skip for layer 0, same as official behavior)
        if self.layer_id > 0 and self.v0 is not None:
            v = v + (torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2))

        # 5. In-context learning rate (a): how much to trust new info
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)

        # 6. Gate (g): output modulation
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        # 7. Key normalization and modulation
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        # 8. Core WKV operator (the O(n) attention replacement)
        # Compute decay: exp(-exp(w)) — matches official RWKV-7 non-CUDA path
        #   Official call: RWKV7_OP(r, w, k, v, -kk, kk*a)
        #   - 5th arg (-kk): key-dependent removal term for Delta Rule
        #   - 6th arg (kk*a): modulated auxiliary term
        w_decay = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
        x = wkv7_forward(
            r.view(B, T, H, N).float(),
            w_decay,
            k.view(B, T, H, N).float(),
            v.view(B, T, H, N).float(),
            (-kk).view(B, T, H, N).float(),         # -kk: removal term (NOT a)
            (kk * a).view(B, T, H, N).float(),      # kk * a: modulated auxiliary
        ).view(B, T, C).to(x.dtype)

        # 9. Layer normalization
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        # 10. Bonus: local attention via r_k
        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
            .sum(dim=-1, keepdim=True)
            * v.view(B, T, H, -1)
        ).view(B, T, C)

        # 11. Output gate
        x = self.output(x * g)
        return x


class RWKV_CMix_x070(nn.Module):
    """RWKV-7 Channel Mixing block (the "FFN" replacement)."""

    def __init__(self, args: RWKV7Config, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.ones(1, 1, args.n_embd))
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

        # Zero-init value projection
        nn.init.zeros_(self.value.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2  # ReLU² activation
        return self.value(k)


class Block(nn.Module):
    """RWKV-7 Block: TimeMix + ChannelMix with residual connections."""

    def __init__(self, args: RWKV7Config, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CourageLM(nn.Module):
    """
    RWKV-7 Courage Language Model.
    
    A 25M-parameter model trained with Digimon "Courage Crest" philosophy
    embedded in its pre-training data.
    
    Architecture:
        Embedding → [Block × n_layer] → LayerNorm → Head
    
    Each Block:
        LayerNorm → TimeMix (WKV attention) → +residual
        LayerNorm → ChannelMix (ReLU² FFN) → +residual
    """

    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.config = config

        # Embedding
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)

        # Special layer-0 pre-norm (RWKV convention)
        self.ln0 = nn.LayerNorm(config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(config, i) for i in range(config.n_layer)
        ])

        # Output
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie embedding and head weights (common in small models)
        self.head.weight = self.emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            idx: token indices of shape (B, T)
        
        Returns:
            logits of shape (B, T, vocab_size)
        """
        B, T = idx.shape

        # Embedding + layer-0 pre-norm
        x = self.ln0(self.emb(idx))

        # Stacked RWKV-7 blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.ln_out(x)
        logits = self.head(x)

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            idx: starting token indices (B, T)
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (1.0 = no change)
        
        Returns:
            generated sequence (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = idx[:, -self.config.ctx_len:]
            # Forward pass (no grad for inference efficiency)
            with torch.no_grad():
                logits = self(idx_cond)
            # Last timestep logits
            logits = logits[:, -1, :] / temperature
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
