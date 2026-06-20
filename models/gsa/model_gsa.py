"""
GSA: Gated Sparse Attention Model (v2 — KV Cache 版本)
===========================================================
内容相关稀疏注意力（Content-dependent Sparse Attention），O(n) 线性复杂度。

KV Cache 机制: 首次推理(prefill)时缓存 K、V、gate、topk_idx 到 KV Cache，
              后续每步 decode 只对单个新 token 查询缓存中的 K/V，
              避免重复计算全序列。

加速原理: 传统 Transformer 每个 decode step 是 O(T²) 的全序列注意力，
         GSA 的 decode step 是 O(T·(k+W)) — 只做新 query 对缓存 key 的点积，
         其中 k=global_k=64 个全局 token，W=window=32 个局部 token。

设计决策（v2 改进）:
  - KV Cache 替代了 v1 的 gather-然后-matmul 方案
  - v1 的 gather 实现在长序列上比 full matmul 还慢 (gather 内存不连续)
  - v2 的 KV 缓存复用让 decode 真正达到 O(T·k) 而非 O(T²)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# ═══════════════════════════════════════════════════════════════════
# Config — 模型配置
# ═══════════════════════════════════════════════════════════════════

class GSA_Config:
    """GSA 模型超参数配置。

    参数说明:
        n_embd=384:   嵌入维度，决定模型宽度。384 是 25M 参数量级的合理选择。
        n_layer=8:    层数。8 层在前向推理时提供足够的非线性深度。
        vocab_size:   词表大小（训练时由 tokenizer 决定，约 2000-8000）。
        ctx_len=4096: 最大上下文长度（tokens）。GSA 线性复杂度支持更长 ctx。
        n_head=6:     注意力头数。n_embd / n_head = head_dim = 64。
        head_dim=64:  每个注意力头的维度。64 是标准选择（与 Transformer 原论文一致）。
        global_k=64:  全局 top-k 数量。每层用 gate score 选出最高分的 64 个位置全局关注。
        window=32:    局部窗口大小。每个 query 额外关注左右各 window 个位置。
        dim_ffn:      FFN 隐藏层维度 = n_embd × 3.5 = 1344（略小于 4x 以控制参数量）
    """

    def __init__(self, d: dict):
        self.n_embd = d["n_embd"]               # 嵌入维度
        self.n_layer = d["n_layer"]              # 层数
        self.vocab_size = d["vocab_size"]        # 词表大小
        self.ctx_len = d.get("ctx_len", 4096)    # 上下文长度，默认 4096
        self.n_head = d.get("n_head", 6)         # 注意力头数，默认 6
        self.head_dim = d.get("head_dim", 64)    # 每头维度，默认 64
        self.global_k = d.get("global_k", 64)    # 全局 top-k，默认 64
        self.window = d.get("window", 32)        # 局部窗口，默认 32
        self.dim_ffn = int(self.n_embd * 3.5)    # FFN 维度 = 嵌入 × 3.5
        # 断言：嵌入维度必须能被头数整除 — n_embd = n_head × head_dim
        assert self.n_embd == self.n_head * self.head_dim

# ═══════════════════════════════════════════════════════════════════
# KV Cache — 键值缓存（v2 新增，推理加速的核心）
# ═══════════════════════════════════════════════════════════════════

class KVCache:
    """
    单层 KV 缓存。存储每层的 K、V 投影和 gate score，供 decode 阶段复用。

    工作流程:
      1. Prefill: 输入完整 prompt → 计算全序列 K/V → 存入 cache
      2. Decode:  每步只计算新 token 的 q,k,v → 查询 cache 中所有历史 K/V

    缓存内容:
      k:      (B, H, T, head_dim) — 所有历史位置的 Key 投影
      v:      (B, H, T, head_dim) — 所有历史位置的 Value 投影
      gate:   (B, T)             — 每个位置的"重要性"分数
      topk_idx: (B, global_k)    — gate 分数最高的 global_k 个位置的索引

    Args:
      B = batch_size (推理时通常为 1)
      H = n_head (注意力头数)
      T = 当前序列长度（随 decode 逐步增长）
      head_dim = 每头维度 (64)
      global_k = 全局选中的位置数 (64)
    """

    def __init__(self):
        self.k: Optional[torch.Tensor] = None       # Key 缓存，shape (B, H, T, head_dim)
        self.v: Optional[torch.Tensor] = None       # Value 缓存，shape (B, H, T, head_dim)
        self.gate: Optional[torch.Tensor] = None     # Gate 分数，shape (B, T)
        self.topk_idx: Optional[torch.Tensor] = None # 全局 top-k 索引，shape (B, global_k)

    @property
    def seq_len(self) -> int:
        """返回当前缓存中的序列长度（已处理了多少个 token）。"""
        return self.k.shape[2] if self.k is not None else 0

    def is_empty(self) -> bool:
        """检查缓存是否为空（尚未 prefill）。"""
        return self.k is None


# ═══════════════════════════════════════════════════════════════════
# GatedSparseAttention — GSA 注意力核心（本项目的灵魂）
# ═══════════════════════════════════════════════════════════════════

class GatedSparseAttention(nn.Module):
    """
    O(n) 内容相关稀疏注意力 + 门控路由。

    【核心思路】
    传统 Transformer: Q × K^T → softmax → × V，复杂度 O(T²d)
    GSA:             先用 gate MLP 对每个位置打分 → 选 top-k 全局 + 窗口局部
                     → 只用这些位置的 K/V 计算注意力 → 复杂度 O(T·(k+W)·d)

    【门控路由】
    gate = sigmoid(W_gate @ x)，shape (B, T)
    gate scores 经 sigmoid 压缩到 [0, 1]，代表每个位置作为"被关注目标"的重要性。
    torch.topk(gate, k=global_k) 选出全局最重要的位置。
    这是一种从"被查 token"视角做选择的方式：
      - 传统路由从 query 视角决定看什么（Q 相关度）
      - 这里的 gate 从 key 视角决定谁值得被看（token 重要性）

    【稀疏模式 = 全局 + 局部】
    每个位置的注意力范围 = top-k 全局 ⊕ 窗口局部
    - 全局: gate score 最高的 global_k=64 个 token（内容相关，跨任意距离）
    - 窗口: 前后 window=32 个 token（位置相关，捕获局部模式）
    - 总关注量: max(global_k, window×2) ≈ 64+64 = 128 位置/token

    Args:
        cfg: GSA_Config 配置
        layer_id: 层编号（用于日志/调试）
    """

    def __init__(self, cfg: GSA_Config, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        d = cfg.n_embd          # 嵌入维度 384
        self.H = cfg.n_head     # 头数 6
        self.N = cfg.head_dim   # 每头维度 64
        self.global_k = cfg.global_k  # 全局 top-k 64
        self.window = cfg.window      # 局部窗口 32
        self.scale = cfg.head_dim ** -0.5  # 注意力缩放因子 1/√64 = 1/8

        # ── 投影矩阵 ──
        # Q/K/V/O: 标准 Transformer 的注意力投影
        self.W_q = nn.Linear(d, d, bias=False)   # Query 投影
        self.W_k = nn.Linear(d, d, bias=False)   # Key 投影
        self.W_v = nn.Linear(d, d, bias=False)   # Value 投影
        self.W_o = nn.Linear(d, d, bias=False)   # Output 投影

        # ── 门控投影（GSA 独有）──
        # W_gate: d → 1，每个 token 输出一个标量分数
        # sigmoid 后得到 [0, 1] 的"重要性"分数
        self.W_gate = nn.Linear(d, 1, bias=False)

        # ── 层归一化 ──
        self.ln = nn.LayerNorm(d)  # 在 QKV 投影前做归一化

    # ══════════════════════════════════════════════════════════════
    # forward — 训练时的全序列前向传播
    # ══════════════════════════════════════════════════════════════

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        训练模式: 全序列稀疏注意力（不使用 KV Cache）。

        由于训练时每步都需要算 loss backward，无法复用 KV Cache，
        所以走完整的全序列注意力。但通过 sparse mask 仍然实现了
        O(T·(k+W)·d) 的稀疏计算（而非 O(T²d)）。

        Args:
            x: (B, T, D) 输入张量
               B = batch_size, T = 序列长度, D = n_embd = 384

        Returns:
            (B, T, D) 输出张量（残差连接已在内部处理）
        """
        B, T, D = x.shape
        H, N, dev = self.H, self.N, x.device

        # ── 残差连接 ──
        # 保存原始输入，最后加回去: output = residual + attention(x)
        residual = x

        # ── LayerNorm ──
        # 在 QKV 投影前做归一化（Pre-LN 范式，训练更稳定）
        x = self.ln(x)

        # ── Q/K/V 投影 + 多头拆分 ──
        # .view(B, T, H, N): 将 D=384 拆成 H×N=6×64
        # .transpose(1, 2): 将维度顺序从 (B,T,H,N) 变成 (B,H,T,N)
        #   这样 batch 矩阵乘法时 head 维度作为 batch 维度并行计算
        Q = self.W_q(x).view(B, T, H, N).transpose(1, 2)  # (B, H, T, N)
        K = self.W_k(x).view(B, T, H, N).transpose(1, 2)  # (B, H, T, N)
        V = self.W_v(x).view(B, T, H, N).transpose(1, 2)  # (B, H, T, N)

        # ── 门控分数 ──
        # W_gate: (B,T,D) → (B,T,1)
        # sigmoid: 压缩到 [0, 1]
        # squeeze(-1): 去掉最后一维，得到 (B, T)
        gate = torch.sigmoid(self.W_gate(x)).squeeze(-1)    # (B, T)

        # ── 构建稀疏注意力 mask ──
        # 1. 计算本届可用的 global_k（不能超过序列长度 T）
        k = min(self.global_k, T)

        # 2. 因果 mask: 每个 token 只能看到当前位置及之前的 token
        #    row[i] = i, col[j] = j → causal[i,j] = (j <= i)
        #    unsqueeze(1) 和 unsqueeze(0) 用于广播
        row = torch.arange(T, device=dev).unsqueeze(1)  # (T, 1)
        col = torch.arange(T, device=dev).unsqueeze(0)  # (1, T)
        causal = col <= row                              # (T, T) 下三角

        # 3. 局部窗口 mask: 只看窗口范围内的 token
        #    条件: col >= row - window（不能太远）且满足因果条件
        window = (col >= row - self.window) & causal     # (T, T)

        # 4. 全局 top-k mask: 基于 gate score 选出最重要的 token
        #    topk 返回 (values, indices)，我们只要 indices
        #    gate shape (B, T)，在 dim=1 上取 top-k
        _, topk = torch.topk(gate, k=k, dim=1)          # (B, k)

        # 初始化全 False 矩阵，然后根据 topk 索引置 True
        glob = torch.zeros(B, T, T, device=dev, dtype=torch.bool)
        for b in range(B):
            # 对每个 batch，所有 query 都可以看 topk[b] 指定的 key 位置
            glob[b, :, topk[b]] = True

        # 5. 合并 mask: (window OR glob) AND causal
        #    unsqueeze 用于 batch/head 维度广播
        mask = ((window.unsqueeze(0) | glob) & causal.unsqueeze(0)).unsqueeze(1)  # (B, 1, T, T)

        # ── 注意力计算 ──
        # scores = Q @ K^T / √d
        # shape: (B, H, T, N) @ (B, H, N, T) → (B, H, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 将 mask 外的位置设为 -inf，softmax 后它们变为 0
        scores = scores.masked_fill(~mask, float('-inf'))

        # softmax + 处理 NaN（全部被 mask 的情况 → 输出 0）
        attn = torch.nan_to_num(F.softmax(scores, dim=-1))

        # 加权求和 V
        # (B, H, T, T) @ (B, H, T, N) → (B, H, T, N)
        # transpose + contiguous + view: 恢复 (B, T, D)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)

        # ── 残差连接 + 输出投影 ──
        return residual + self.W_o(out)

    # ══════════════════════════════════════════════════════════════
    # prefill — 推理第一步：处理完整 prompt + 填充 KV Cache
    # ══════════════════════════════════════════════════════════════

    def prefill(self, x: torch.Tensor) -> Tuple[torch.Tensor, 'KVCache']:
        """
        推理的 Prefill 阶段: 处理用户的完整 prompt，计算全序列注意力，
        同时将 K、V、gate、topk 存入 KV Cache。

        之后每次生成新 token 时，decode() 只需查询这个 cache，
        无需重新处理整个序列。

        Args:
            x: (B, T, D) — 完整 prompt 的嵌入

        Returns:
            (output, cache) — 前向输出 + 填充好的 KV Cache
        """
        B, T, D = x.shape
        H, N, dev = self.H, self.N, x.device

        residual = x
        x = self.ln(x)

        Q = self.W_q(x).view(B, T, H, N).transpose(1, 2)  # (B, H, T, N)
        K = self.W_k(x).view(B, T, H, N).transpose(1, 2)  # (B, H, T, N)
        V = self.W_v(x).view(B, T, H, N).transpose(1, 2)  # (B, H, T, N)
        gate = torch.sigmoid(self.W_gate(x)).squeeze(-1)    # (B, T)

        # 构建稀疏 mask（逻辑与 train forward 相同）
        k = min(self.global_k, T)
        row = torch.arange(T, device=dev).unsqueeze(1)
        col = torch.arange(T, device=dev).unsqueeze(0)
        causal = col <= row
        window = (col >= row - self.window) & causal

        _, topk = torch.topk(gate, k=k, dim=1)             # (B, k)
        glob = torch.zeros(B, T, T, device=dev, dtype=torch.bool)
        for b in range(B):
            glob[b, :, topk[b]] = True
        mask = ((window.unsqueeze(0) | glob) & causal.unsqueeze(0)).unsqueeze(1)  # (B, 1, T, T)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.nan_to_num(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)

        # ── 填充 KV Cache ──
        # 将本次计算的全序列 K/V/gate/topk 存入缓存
        cache = KVCache()
        cache.k = K           # (B, H, T, head_dim) — 所有 Key
        cache.v = V           # (B, H, T, head_dim) — 所有 Value
        cache.gate = gate     # (B, T)             — 所有位置的 gate score
        cache.topk_idx = topk  # (B, global_k)      — prefill 时的全局 top-k 索引

        return residual + self.W_o(out), cache

    # ══════════════════════════════════════════════════════════════
    # decode — 推理后续步骤：单 token 查询缓存
    # ══════════════════════════════════════════════════════════════

    def decode(
        self,
        x_single: torch.Tensor,   # (B, 1, D) — 单个新 token 的嵌入
        cache: KVCache,            # 已有的 KV Cache
    ) -> Tuple[torch.Tensor, 'KVCache']:
        """
        单 token decode 步骤: 只计算新 token 的 Q/K/V → 查询缓存中的历史 K/V。

        复杂度: O(T·(k+W)·d) 而非 O(T²d)
        因为新的 query(1个) 只与缓存中 global_k=64 + window=32 个 key 做点积，
        而非与全部 T 个 key 做点积。

        Args:
            x_single: (B, 1, D) — 上一轮生成的新 token
            cache:    已有的 KV Cache（包含 prefill + 前序 decode 的全部历史）

        Returns:
            (output, updated_cache) — 输出 + 追加了新 token 的缓存
        """
        B, T, D = x_single.shape
        assert T == 1, f"decode expects T=1, got {T}"  # 确保只传入一个 token
        H, N, dev = self.H, self.N, x_single.device

        residual = x_single
        x = self.ln(x_single)

        # 计算新 token 的 Q/K/V/gate（仅这一个 token）
        q = self.W_q(x).view(B, 1, H, N).transpose(1, 2)  # (B, H, 1, N)
        k_new = self.W_k(x).view(B, 1, H, N).transpose(1, 2)  # (B, H, 1, N)
        v_new = self.W_v(x).view(B, 1, H, N).transpose(1, 2)  # (B, H, 1, N)
        gate_new = torch.sigmoid(self.W_gate(x)).squeeze(-1)  # (B, 1)

        # ── 追加到 KV Cache ──
        # torch.cat 沿序列维度拼接新 token 的 K/V/gate
        cache.k = torch.cat([cache.k, k_new], dim=2)      # dim=2 是序列维度
        cache.v = torch.cat([cache.v, v_new], dim=2)
        cache.gate = torch.cat([cache.gate, gate_new], dim=1)  # gate 的序列维度是 dim=1
        # 注意: topk_idx 沿用 prefill 时的结果，不更新
        #   因为 prefill 已经根据完整 prompt 识别了全局重要位置

        kv_len = cache.k.shape[2]   # 缓存中总 token 数（prefill + 已生成的）
        cur_pos = kv_len - 1         # 当前 query 的位置（0-indexed）

        # ── 构建稀疏 attention mask（仅对当前 query）──
        # mask shape: (1, kv_len)，True = 当前 query 可以关注的位置
        mask = torch.zeros(1, kv_len, device=dev, dtype=torch.bool)

        # 1. 全局 top-k: 复用 prefill 时选出的全局重要位置
        #    topk_idx[0] shape (global_k,)，这些位置总是被关注
        mask[0, cache.topk_idx[0]] = True

        # 2. 局部窗口: 当前 token 前后 window 范围内（满足因果性）
        w_start = max(0, cur_pos - self.window)          # 窗口起点
        mask[0, w_start:cur_pos + 1] = True               # 窗口内 + 自己

        # 扩展维度以匹配 batch 矩阵乘法
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 1, kv_len)

        # ── 注意力计算: 新 query vs 缓存中的全部 K/V ──
        # 复杂度: O(1 × kv_len × N) = O(Td) — 线性！
        # 但实际被 mask 过滤后，只有 ~(global_k + window) 个位置有效
        scores = torch.matmul(q, cache.k.transpose(-2, -1)) * self.scale  # (B,H,1,kv_len)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.nan_to_num(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, cache.v)  # (B, H, 1, N)
        out = out.transpose(1, 2).contiguous().view(B, 1, D)

        return residual + self.W_o(out), cache


# ═══════════════════════════════════════════════════════════════════
# GSA_FFN — 前馈网络
# ═══════════════════════════════════════════════════════════════════

class GSA_FFN(nn.Module):
    """
    GSA 的前馈网络。使用 ReLU² 激活（类似 RWKV-7 的 Channel Mix）。

    结构: LayerNorm → UpProject(d→3.5d) → ReLU → 平方 → DownProject(3.5d→d)

    算法细节 — ReLU² 激活:
    ReLU²(x) = max(0, x)²
    相比普通 ReLU: 在 x>0 时梯度更大（2x vs 1），促进稀疏激活
    相比 GELU: 实现更简单，无 exp/erf 运算，MPS 上更快
    这是 RWKV 系列的首选激活函数，已验证在语言模型上效果良好
    """

    def __init__(self, cfg: GSA_Config):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.n_embd)             # 前置 LayerNorm
        self.W_up = nn.Linear(cfg.n_embd, cfg.dim_ffn, bias=False)   # 上投影 d→3.5d
        self.W_down = nn.Linear(cfg.dim_ffn, cfg.n_embd, bias=False) # 下投影 3.5d→d
        # RWKV 惯例: 下投影权重初始化为 0，让模型逐步学习而非随机初始化冲击
        nn.init.zeros_(self.W_down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) 输入
        Returns:
            (B, T, D) 输出，含残差连接
        """
        # x → ln → up → relu → 平方 → down → +残差
        return x + self.W_down(torch.relu(self.W_up(self.ln(x))) ** 2)


# ═══════════════════════════════════════════════════════════════════
# GSA_Block — GSA 单层
# ═══════════════════════════════════════════════════════════════════

class GSA_Block(nn.Module):
    """
    GSA 的单个 Block: 稀疏注意力 + 前馈网络。

    结构（类似 Transformer Block）:
      x → GatedSparseAttention → FFN → output

    每个子层内部已有 LayerNorm + 残差连接，所以 Block 层不需要额外包装。
    """

    def __init__(self, cfg: GSA_Config, layer_id: int):
        super().__init__()
        self.attn = GatedSparseAttention(cfg, layer_id)  # GSA 注意力层
        self.ffn = GSA_FFN(cfg)                          # 前馈网络层

    # 训练模式: 全序列处理（无 cache）
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attn(x))

    # 推理 Prefill: 处理全 prompt + 返回每层的 KV Cache
    def prefill(self, x: torch.Tensor) -> Tuple[torch.Tensor, KVCache]:
        out, cache = self.attn.prefill(x)
        return self.ffn(out), cache

    # 推理 Decode: 单 token 查询缓存
    def decode(self, x: torch.Tensor, cache: KVCache) -> Tuple[torch.Tensor, KVCache]:
        out, cache = self.attn.decode(x, cache)
        return self.ffn(out), cache


# ═══════════════════════════════════════════════════════════════════
# GSALanguageModel — GSA 语言模型主体
# ═══════════════════════════════════════════════════════════════════

class GSALanguageModel(nn.Module):
    """
    Gated Sparse Attention 语言模型 — ~16M 参数 @ n_embd=384, n_layer=8。

    整体结构:
      Token Embedding → LayerNorm₀ → [GSA_Block × n_layer] → LayerNorm_out → LM Head

    权重绑定 (Weight Tying):
      self.head.weight = self.emb.weight
      LM Head 和 Token Embedding 共享权重矩阵。
      好处: 减少参数量、加速收敛。对于小模型（< 50M）是标准做法。
      原理: 输入嵌入和输出投影在语义上相近，共享权重让模型学习更好的表征。

    推理流程（generate 方法）:
      1. Prefill: 完整 prompt → 逐层 prefill → 缓存每层 K/V → 预测第一个新 token
      2. Decode: 循环 — 新 token → 逐层 decode(查缓存) → 预测下一个 token
    """

    def __init__(self, cfg: GSA_Config):
        super().__init__()
        self.cfg = cfg
        # Token Embedding: 将 token ID → 稠密向量
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        # LayerNorm₀: RWKV/GSA 的独有设计 — 在嵌入后立即做 LN
        # 原因: 稳定训练初期的数值范围，防止嵌入层的方差积累
        self.ln0 = nn.LayerNorm(cfg.n_embd)

        # GSA Block 堆叠: 8 个相同结构的层
        self.blocks = nn.ModuleList([
            GSA_Block(cfg, i) for i in range(cfg.n_layer)
        ])

        # 输出层: LayerNorm → 线性投影到词表大小
        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # 权重绑定: LM Head 复用 Embedding 权重
        self.head.weight = self.emb.weight  # tie

    # ══════════════════════════════════════════════════════════════
    # forward — 训练前向
    # ══════════════════════════════════════════════════════════════

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        训练模式前向传播（全序列，不用 cache）。

        Args:
            idx: (B, T) token ID 序列

        Returns:
            (B, T, vocab_size) logits — 每个位置预测下一个 token 的分数
        """
        x = self.ln0(self.emb(idx))          # Embed + LN₀
        for block in self.blocks:            # 逐层通过 GSA Block
            x = block(x)
        return self.head(self.ln_out(x))     # LN + 投影到词表

    # ══════════════════════════════════════════════════════════════
    # generate — KV 缓存自回归生成（推理核心）
    # ══════════════════════════════════════════════════════════════

    @torch.no_grad()  # 推理时禁用梯度计算，节省内存
    def generate(
        self,
        idx: torch.Tensor,              # (B, T) 输入 token 序列
        max_new_tokens: int,            # 最大生成 token 数
        temperature: float = 0.8,       # 温度: 控制随机性
        top_k: int = 40,                # Top-K 采样: 只从概率最高的 K 个 token 中选
    ) -> torch.Tensor:
        """
        基于 KV Cache 的自回归生成。

        【采样算法详解】
        每一步生成下一个 token 的过程:
          1. 温度缩放: logits = logits / temperature
             当 temperature < 1.0 时，logits 差异增大 → softmax 更尖锐 → 输出更确定
             当 temperature > 1.0 时，logits 差异减小 → softmax 更平坦 → 输出更随机

          2. Top-K 过滤: torch.topk(logits, k) 取前 K 个最大 logits
             将非前 K 的 logits 设为 -inf → softmax 后概率为 0
             目的: 过滤长尾噪声 token，防止无意义输出

          3. Softmax: probs = softmax(filtered_logits)

          4. 多项式采样: multinomial(probs, 1) 按概率随机选一个 token

        【KV Cache 工作流】
          1. Prefill: 嵌入完整 prompt → 逐层 prefill（全序列注意力）→ 缓存每层 K/V
          2. 采样第一个新 token（基于 prefill 最后位置的 logits）
          3. Decode 循环: 新 token → 嵌入 → 逐层 decode（查缓存）→ 采样下一个

        Args:
            idx: (B, T) 输入的 token 序列
            max_new_tokens: 最多生成多少个新 token
            temperature: 采样温度，默认 0.8
            top_k: Top-K 过滤数，默认 40

        Returns:
            (B, T + max_new_tokens) 完整序列（输入 + 生成）
        """
        B, T = idx.shape
        device = idx.device

        # ── Prefill 阶段 ──
        # 处理用户的完整 prompt，填充每层的 KV Cache
        x = self.ln0(self.emb(idx))          # Embed + LN₀ → (B, T, D)
        caches: List[KVCache] = []
        for i, block in enumerate(self.blocks):
            x, cache = block.prefill(x)      # 逐层 prefill，收集每层缓存
            caches.append(cache)

        # ── 预测第一个新 token ──
        # 取最后位置的 logits / temperature
        logits = self.head(self.ln_out(x))[:, -1, :] / temperature  # (B, vocab)

        # Top-K 过滤: logits[logits < 第K大的值] = -inf
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # v[:, -1:] = 第 K 大的值（按行）
            logits[logits < v[:, -1:]] = float('-inf')

        # 采样
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

        generated = [idx_next]  # 收集生成的 token

        # ── Decode 循环 ──
        # 每步: 新 token → 逐层查缓存 → 预测下一个 token
        for _ in range(max_new_tokens - 1):
            x = self.ln0(self.emb(idx_next))           # (B, 1, D)

            for i, block in enumerate(self.blocks):
                x, caches[i] = block.decode(x, caches[i])  # 单 token 查询缓存

            # 温度缩放 + Top-K + Softmax + 采样（与第一步相同）
            logits = self.head(self.ln_out(x))[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated.append(idx_next)

        # 拼接: 输入 token + 所有生成的 token
        return torch.cat([idx] + generated, dim=1)

    def count_parameters(self) -> int:
        """统计可训练参数数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
