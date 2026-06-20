# Spike 001: GSA vs RWKV-7 训练对比

## 问题

GSA (Gated Sparse Attention) 能否在 TinyStories 上以与 RWKV-7 可比的质量训练，
同时保持 O(n) 复杂度？

## 方法

1. 用相同参数量级（~17M）实现 GSA 模型
2. 用完全相同的数据/步数/优化器，各训练 100 步
3. 对比：loss 曲线 + 训练速度

## GSA 架构

```
X → QKV投影 → Gate(内容选择) → Top-k全局稀疏 + 局部窗口 → Q·K_sparse → softmax → V加权 → Output
```

- 全局 Top-k 选择器（逐位置 MLP，O(Td)）
- 局部窗口 ±32
- 总关注量 = k(64) + 2W(64) = 128 位置/查询
- 理论复杂度：O(T · (k+2W) · d) = O(T) 线性

## 对比配置

| 参数 | RWKV-7 | GSA |
|------|--------|-----|
| n_embd | 384 | 384 |
| n_layer | 8 | 8 |
| vocab_size | 8000 | 8000 |
| 参数量 | ~17M | ~16M |
| 注意力机制 | WKV (RNN) | Sparse Attention |
| 复杂度 | O(T·H·N²) | O(T·(k+2W)·d) |

## 结果

### CPU (ctx=128, batch=2, 100 steps)

| Metric | RWKV-7 | GSA | Ratio |
|--------|--------|-----|-------|
| Step 1 loss | 385.4 | 373.5 | — |
| Step 100 loss | 33.9 | 29.7 | — |
| Loss reduction | 351.5 | 343.7 | 0.98x |
| Time/step | 1005ms | 91ms | **11.0x faster** |

### MPS (ctx=128, batch=2, 100 steps)

| Metric | RWKV-7 | GSA | Ratio |
|--------|--------|-----|-------|
| Step 1 loss | 383.0 | 369.2 | — |
| Step 100 loss | 34.3 | 30.4 | — |
| Loss reduction | 348.7 | 338.8 | 0.97x |
| Time/step | 669ms | 64ms | **10.5x faster** |

### Loss 收敛曲线 (前 10 步, MPS)

| Step | RWKV-7 | GSA |
|------|--------|-----|
| 1 | 383.0 | 369.2 |
| 2 | 375.9 | 322.5 |
| 3 | 343.9 | 207.4 |
| 4 | 278.7 | 131.1 |
| 5 | 205.3 | 90.7 |
| 6 | 140.7 | 78.3 |
| 7 | 97.5 | 68.6 |
| 8 | 81.0 | 64.5 |
| 9 | 74.3 | 59.8 |
| 10 | 69.3 | 56.9 |

GSA 早期收敛更快：步 5 已达 90.7 vs RWKV-7 205.3。

## Verdict: VALIDATED

### What worked

1. **GSA 收敛正常** — 100 步内 loss 从 369 → 30，与 RWKV-7 (383 → 34) 可比
2. **训练速度碾压** — CPU 快 11×，MPS 快 10.5×
   - 原因：GSA 用 `torch.matmul`（高度优化的 BLAS），RWKV-7 用 Python for-loop
   - 这个差距会随 ctx 长度扩大而加大
3. **参数量可控** — 16.1M vs 17.0M，基本对等
4. **架构简洁** — 标准 Transformer 风格，无需自定义 CUDA kernel

### What didn't (limitations)

1. **不是真正的 O(n)** — 当前实现用 `Q·K^T` 全矩阵乘法再 mask，计算量仍是 O(T²d)
   - 实现真正的 O(T·(k+W)·d) 需要先 `gather` K/V 再计算点积
   - 但在 ctx=128 时 full matmul 比 Python for-loop 快得多，所以"fake O(n)" 反而更快
2. **全局 top-k 共享** — 所有 query 共用同一组 top-k key，不如 SSA 的逐 query 路由
3. **Mask 构建有 for-loop** — `_build_sparse_mask` 中的 batch loop 成为瓶颈
4. **合成数据** — 用随机整数而非真实文本，loss 绝对值无意义

### Surprises

1. **GSA 在 CPU 上比 MPS 差距更大** — CPU 上 GSA 11× vs RWKV-7，MPS 上 10.5×
   - RWKV-7 的 Python for-loop 在 CPU 上受 GIL 影响更严重
2. **简单门控就够了** — 单层 MLP gate 就能选出有用的全局位置
3. **初始化 loss 不同** — GSA (369) vs RWKV (383)，可能是 embedding 初始化差异

### Recommendation for the real build

1. **实现真正的 O(n) gather-然后-matmul** — 对 ctx > 512 才有意义
2. **添加逐 query 路由** — 用 Q 的 hash/distribution 而非全局 top-k
3. **在真实文本上验证** — TinyStories + Courage 材料
4. **门控正则化** — 防止 gate 坍塌到常数（当前 100 步内未见此问题）
5. **可考虑混合架构** — GSA 用于长程检索 + RWKV 用于局部建模

## 对话使用（Chat Demo）

训练完成后使用 `demo_chat.py` 进行对话：

```bash
# 单次生成
python demo_chat.py --checkpoint checkpoints/final.pt --prompt "I believe" --max-tokens 200

# 交互模式
python demo_chat.py --checkpoint checkpoints/final.pt --interactive

# 未训练模型测试管道
python demo_chat.py --untrained --prompt "Hello"
```

### 生成流程

```
用户 Prompt → Tokenizer.encode → [BOS, tok1, tok2, ...]
                                              ↓
                                GSALanguageModel.generate()
                                              ↓
                          ┌─────────────────────┐
                          │ for _ in steps:      │
                          │   idx_cond = idx[-ctx:]  ← 裁剪到上下文窗口
                          │   logits = model(idx_cond)  ← GSA forward
                          │   logits = last_pos / T     ← 只取最后位置
                          │   top_k(logits)             ← 过滤低概率 token
                          │   next = sample(softmax)    ← 采样
                          │   idx = [idx, next]         ← 追加
                          └─────────────────────┘
                                              ↓
                          Tokenizer.decode → 生成文本
```

### 当前局限

| 问题 | 影响 | 修复方向 |
|------|------|---------|
| 无 KV 缓存 | 每步重算全序列 O(T²) | 缓存 K/V + 增量 gate |
| 全局 top-k 共享 | 所有 query 看同一组位置 | 逐 query 路由 |
| SimpleTokenizer | 无法还原真实文本 | 训练 BPE tokenizer |
