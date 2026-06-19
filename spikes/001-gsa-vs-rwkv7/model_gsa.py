"""
GSA: Gated Sparse Attention Model (v2 — KV Cache)
===================================================
Content-dependent sparse attention, O(n) complexity.

KV Cache: prefill caches K/V/gate/topk → decode steps reuse cache.
Speedup: O(T²) → O(T·(k+W)) per decode step (only new query vs cache).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ─── Config ───────────────────────────────────────────────────────────────

class GSA_Config:
    def __init__(self, d: dict):
        self.n_embd = d["n_embd"]
        self.n_layer = d["n_layer"]
        self.vocab_size = d["vocab_size"]
        self.ctx_len = d.get("ctx_len", 4096)
        self.n_head = d.get("n_head", 6)
        self.head_dim = d.get("head_dim", 64)
        self.global_k = d.get("global_k", 64)
        self.window = d.get("window", 32)
        self.dim_ffn = int(self.n_embd * 3.5)
        assert self.n_embd == self.n_head * self.head_dim


# ─── KV Cache ─────────────────────────────────────────────────────────────

class KVCache:
    """Per-layer cache: K, V, gate scores, global top-k indices."""

    def __init__(self):
        self.k: Optional[torch.Tensor] = None       # (B, H, T, head_dim)
        self.v: Optional[torch.Tensor] = None       # (B, H, T, head_dim)
        self.gate: Optional[torch.Tensor] = None     # (B, T)
        self.topk_idx: Optional[torch.Tensor] = None # (B, global_k)

    @property
    def seq_len(self) -> int:
        return self.k.shape[2] if self.k is not None else 0

    def is_empty(self) -> bool:
        return self.k is None


# ─── GSA Attention Block ──────────────────────────────────────────────────

class GatedSparseAttention(nn.Module):
    """O(n) sparse attention with content-dependent gate routing."""

    def __init__(self, cfg: GSA_Config, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        d = cfg.n_embd
        self.H = cfg.n_head
        self.N = cfg.head_dim
        self.global_k = cfg.global_k
        self.window = cfg.window
        self.scale = cfg.head_dim ** -0.5

        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.W_gate = nn.Linear(d, 1, bias=False)
        self.ln = nn.LayerNorm(d)

    # ── Training forward ──────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training: full-sequence sparse attention (no cache)."""
        B, T, D = x.shape
        H, N, dev = self.H, self.N, x.device

        residual = x
        x = self.ln(x)

        Q = self.W_q(x).view(B, T, H, N).transpose(1, 2)  # (B,H,T,N)
        K = self.W_k(x).view(B, T, H, N).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, N).transpose(1, 2)
        gate = torch.sigmoid(self.W_gate(x)).squeeze(-1)    # (B,T)

        # Build sparse mask: causal & (window | global_topk)
        k = min(self.global_k, T)
        row = torch.arange(T, device=dev).unsqueeze(1)
        col = torch.arange(T, device=dev).unsqueeze(0)
        causal = col <= row
        window = (col >= row - self.window) & causal

        _, topk = torch.topk(gate, k=k, dim=1)  # (B, k)
        glob = torch.zeros(B, T, T, device=dev, dtype=torch.bool)
        for b in range(B):
            glob[b, :, topk[b]] = True

        mask = ((window.unsqueeze(0) | glob) & causal.unsqueeze(0)).unsqueeze(1)  # (B,1,T,T)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.nan_to_num(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)

        return residual + self.W_o(out)

    # ── Inference: prefill ────────────────────────────────────────────

    def prefill(self, x: torch.Tensor) -> Tuple[torch.Tensor, 'KVCache']:
        """First call: full attention + populate cache. Returns (output, cache)."""
        B, T, D = x.shape
        H, N, dev = self.H, self.N, x.device

        residual = x
        x = self.ln(x)

        Q = self.W_q(x).view(B, T, H, N).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, N).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, N).transpose(1, 2)
        gate = torch.sigmoid(self.W_gate(x)).squeeze(-1)

        # Build sparse mask
        k = min(self.global_k, T)
        row = torch.arange(T, device=dev).unsqueeze(1)
        col = torch.arange(T, device=dev).unsqueeze(0)
        causal = col <= row
        window = (col >= row - self.window) & causal

        _, topk = torch.topk(gate, k=k, dim=1)
        glob = torch.zeros(B, T, T, device=dev, dtype=torch.bool)
        for b in range(B):
            glob[b, :, topk[b]] = True
        mask = ((window.unsqueeze(0) | glob) & causal.unsqueeze(0)).unsqueeze(1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.nan_to_num(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)

        # Populate cache
        cache = KVCache()
        cache.k = K
        cache.v = V
        cache.gate = gate
        cache.topk_idx = topk

        return residual + self.W_o(out), cache

    # ── Inference: decode step ────────────────────────────────────────

    def decode(
        self,
        x_single: torch.Tensor,
        cache: KVCache,
    ) -> Tuple[torch.Tensor, 'KVCache']:
        """
        Single-token decode: query against cached K/V.

        Args:
            x_single: (B, 1, D) — single token embedding
            cache: existing KVCache from prefill + previous decodes
        Returns:
            (output, updated_cache)
        """
        B, T, D = x_single.shape
        assert T == 1, f"decode expects T=1, got {T}"
        H, N, dev = self.H, self.N, x_single.device

        residual = x_single
        x = self.ln(x_single)

        q = self.W_q(x).view(B, 1, H, N).transpose(1, 2)  # (B, H, 1, N)
        k_new = self.W_k(x).view(B, 1, H, N).transpose(1, 2)
        v_new = self.W_v(x).view(B, 1, H, N).transpose(1, 2)
        gate_new = torch.sigmoid(self.W_gate(x)).squeeze(-1)  # (B, 1)

        # Append to cache
        cache.k = torch.cat([cache.k, k_new], dim=2)
        cache.v = torch.cat([cache.v, v_new], dim=2)
        cache.gate = torch.cat([cache.gate, gate_new], dim=1)

        kv_len = cache.k.shape[2]  # total cached positions
        cur_pos = kv_len - 1        # 0-indexed position of this query

        # Build sparse mask for this query
        mask = torch.zeros(1, kv_len, device=dev, dtype=torch.bool)

        # Global top-k (reuse prefill indices)
        mask[0, cache.topk_idx[0]] = True

        # Local window (causal: can only see up to cur_pos)
        w_start = max(0, cur_pos - self.window)
        mask[0, w_start:cur_pos + 1] = True

        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 1, kv_len)

        # Attention against full cached K,V
        scores = torch.matmul(q, cache.k.transpose(-2, -1)) * self.scale  # (B,H,1,kv_len)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.nan_to_num(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, cache.v)  # (B, H, 1, N)
        out = out.transpose(1, 2).contiguous().view(B, 1, D)

        return residual + self.W_o(out), cache


# ─── FFN ──────────────────────────────────────────────────────────────────

class GSA_FFN(nn.Module):
    def __init__(self, cfg: GSA_Config):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.W_up = nn.Linear(cfg.n_embd, cfg.dim_ffn, bias=False)
        self.W_down = nn.Linear(cfg.dim_ffn, cfg.n_embd, bias=False)
        nn.init.zeros_(self.W_down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.W_down(torch.relu(self.W_up(self.ln(x))) ** 2)


# ─── Block ────────────────────────────────────────────────────────────────

class GSA_Block(nn.Module):
    def __init__(self, cfg: GSA_Config, layer_id: int):
        super().__init__()
        self.attn = GatedSparseAttention(cfg, layer_id)
        self.ffn = GSA_FFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attn(x))

    def prefill(self, x: torch.Tensor) -> Tuple[torch.Tensor, KVCache]:
        out, cache = self.attn.prefill(x)
        return self.ffn(out), cache

    def decode(self, x: torch.Tensor, cache: KVCache) -> Tuple[torch.Tensor, KVCache]:
        out, cache = self.attn.decode(x, cache)
        return self.ffn(out), cache


# ─── Language Model ───────────────────────────────────────────────────────

class GSALanguageModel(nn.Module):
    """Gated Sparse Attention LM — 16M params @ n_embd=384, n_layer=8."""

    def __init__(self, cfg: GSA_Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.ln0 = nn.LayerNorm(cfg.n_embd)
        self.blocks = nn.ModuleList([
            GSA_Block(cfg, i) for i in range(cfg.n_layer)
        ])
        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight  # tie

    # ── Training ──────────────────────────────────────────────────────

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.ln0(self.emb(idx))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_out(x))

    # ── Inference (KV-cached) ────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """
        KV-cached autoregressive generation.

        Prefill: embed full prompt → prefill all blocks → cache K/V/gate.
        Decode: each new token → embed → decode all blocks (single-token vs cache).
        """
        B, T = idx.shape
        device = idx.device

        # --- Prefill ---
        x = self.ln0(self.emb(idx))  # (B, T, D)
        caches: List[KVCache] = []
        for i, block in enumerate(self.blocks):
            x, cache = block.prefill(x)
            caches.append(cache)

        logits = self.head(self.ln_out(x))[:, -1, :] / temperature  # (B, vocab)

        # Sample first token
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

        generated = [idx_next]

        # --- Decode loop ---
        for _ in range(max_new_tokens - 1):
            x = self.ln0(self.emb(idx_next))  # (B, 1, D)
            for i, block in enumerate(self.blocks):
                x, caches[i] = block.decode(x, caches[i])

            logits = self.head(self.ln_out(x))[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated.append(idx_next)

        return torch.cat([idx] + generated, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
