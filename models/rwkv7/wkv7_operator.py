"""
RWKV-7 WKV 算子（纯 PyTorch 实现，无 CUDA 依赖）
========================================================
改编自 BlinkDL/RWKV-LM 仓库中的 RWKV-v7/rwkv_v7_demo.py（非 CUDA 路径，第 170-203 行）

WKV 算子实现了 Delta Rule（增量规则）状态更新算法：

    状态更新公式:
        state_t = state_{t-1} * decay_t          # 衰减旧状态（遗忘门）
                + state_{t-1} @ a_t @ b_t        # Delta Rule：选择性移除旧信息
                + v_t @ k_t                      # 建立新的键值关联（学习新信息）

    输出公式:
        output_t = state_t @ r_t                  # receptance 门控查询状态

这是核心的 O(n) 线性注意力机制，用以替代 Transformer 的 O(n²) softmax 注意力。
RWKV-7 的 key 创新在于 Delta Rule：它不仅仅衰减旧状态，而是通过可学习的
a/b 向量主动选择性地从状态矩阵中 "删除" 特定信息，实现更细粒度的上下文更新。

数学背景——Delta Rule（增量规则）:
    Delta Rule 源自计算神经科学中的信用分配（credit assignment）理论。
    在 RWKV-7 中，它将隐藏状态建模为一个可微的联想记忆矩阵：
    - 状态矩阵 S ∈ R^{N×N} 存储了键值对的关联信息
    - a @ b 形成一个秩-1 更新矩阵，用于从记忆中擦除过时信息
    - v @ k 形成另一个秩-1 更新矩阵，用于写入新的键值关联
    - decay 提供指数衰减，使旧信息随时间逐渐淡忘
    三项共同作用，使模型在保持 O(n) 复杂度的同时，具备了选择性记忆与遗忘能力。

许可证: Apache 2.0（继承自 RWKV-LM）
"""

import torch


def wkv7_forward(
    r: torch.Tensor,  # (B, T, H, N) - receptance（接受度门控向量）
    w: torch.Tensor,  # (B, T, H, N) - 衰减权重，已预处理：w = exp(-exp(w_raw))
    k: torch.Tensor,  # (B, T, H, N) - key（键向量）
    v: torch.Tensor,  # (B, T, H, N) - value（值向量）
    a: torch.Tensor,  # (B, T, H, N) - Delta Rule 学习率（"a" 张量，控制移除强度）
    b: torch.Tensor,  # (B, T, H, N) - 辅助张量（与 a 配对，决定移除方向）
) -> torch.Tensor:
    """
    RWKV-7 WKV 算子的纯 PyTorch 前向传播实现。

    这是官方 RWKV-7 演示中的精确非 CUDA 路径实现。
    可在 CPU、MPS 和 CUDA 上运行（尽管 CUDA 原生内核更快）。

    对于 25M 参数量级、4K 上下文的模型，Python 循环的开销
    相对于 O(n) 复杂度所节省的 FLOPs 来说可以忽略不计。

    核心算法——Delta Rule 状态更新:

    每个时间步 t，状态矩阵 state (B, H, N, N) 经历三步更新：

    (1) 指数衰减（遗忘）:
        state ← state * w_t
        w_t = exp(-exp(w_raw_t)) 确保 w_t ∈ (0, 1]
        效果：所有历史信息按通道独立衰减，越久远的信息贡献越小

    (2) Delta Rule（选择性移除）:
        state ← state + state @ a_t @ b_t
        a_t ∈ R^{N×1},  b_t ∈ R^{1×N}  →  a_t @ b_t ∈ R^{N×N}（秩-1 矩阵）
        效果：从状态中主动删除与当前上下文不再相关的旧联想
        这是 RWKV-7 区别于前代的核心创新——不仅仅是被动衰减，
        而是通过可学习的 a/b 向量实现定向信息擦除

    (3) 新键值关联（学习）:
        state ← state + v_t @ k_t
        v_t ∈ R^{N×1},  k_t ∈ R^{1×N}  →  v_t @ k_t ∈ R^{N×N}（秩-1 矩阵）
        效果：将当前 token 的键值关联写入状态矩阵

    输出: out_t = state_t @ r_t
        用 receptance 向量 r_t 查询更新后的状态，控制信息流出

    Dimension 说明:
        B - batch size（批次大小）
        T - sequence length / time steps（序列长度 / 时间步数）
        H - number of heads（注意力头数）
        N - head dimension / state size（每头维度 / 状态矩阵大小）

    Args:
        r: receptance 向量，形状 (B, T, H, N)，控制输出的信息门控
        w: 衰减权重，形状 (B, T, H, N)，已通过 exp(-exp(w_raw)) 变换
           使得值在 (0, 1] 之间，确保稳定衰减
        k: 键向量，形状 (B, T, H, N)
        v: 值向量，形状 (B, T, H, N)
        a: Delta Rule 的学习率/强度，形状 (B, T, H, N)，控制移除信息的幅度
        b: Delta Rule 的辅助向量，形状 (B, T, H, N)，与 a 共同决定移除方向

    Returns:
        输出张量，形状 (B, T, H, N)，每个时间步的注意力输出
    """
    # 解包输入张量的维度
    # B: batch size, T: 序列长度, H: 头数, N: 每头状态维度
    B, T, H, N = r.shape

    # --- 数据类型转换与数值稳定化 ---
    # 将所有权重转换为 float32 以确保数值稳定性
    # 原因：MPS (Apple Silicon GPU) 后端在某些 matmul 操作中
    # 使用 float16 或 bfloat16 时可能出现数值不稳定
    # float32 提供更高的精度，避免 Delta Rule 更新中的累积误差
    r = r.float()
    k = k.float()
    v = v.float()
    a = a.float()
    b = b.float()
    w = w.float()

    # 预分配输出张量：形状与输入一致 (B, T, H, N)
    # 使用 float32 保持一致的数值精度
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)

    # 初始化隐藏状态矩阵：形状 (B, H, N, N)
    # 这是 RWKV 的核心——一个 N×N 的可学习状态矩阵
    # 不同于 Transformer 的 KV cache（显式存储所有历史 K/V 对），
    # RWKV 将历史信息压缩进这个固定大小的矩阵中，实现 O(n) 复杂度
    #
    # state[b, h, :, :] 维护了第 b 个 batch、第 h 个头的键值关联
    # 每一行对应一个 "值维度"，每一列对应一个 "键维度"
    # 矩阵乘法 state @ k 实现高效的向量化记忆检索
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

    # --- 时间步循环：按顺序处理每个 token ---
    # 这必须是一个串行循环，因为 state_t 依赖 state_{t-1}
    # 与 Transformer 的并行注意力不同，RWKV 以循环方式处理序列
    # 这使其天然支持推理时的流式解码（streaming inference）
    for t in range(T):
        # ========== 步骤 1：准备当前时间步的向量 ==========
        # 将所有向量 reshape 为适合批量矩阵乘法的形状
        # 约定：矩阵乘法中，形如 (B, H, M, N) 的张量
        # 前两维 (B, H) 自动广播/批处理，实际计算在 (M, N) 上进行

        # key 向量： (B, H, N) → (B, H, 1, N)
        # 含义：形状变为 "1 行 N 列" 的行向量，用于与 value 做外积
        # kk @ vv 得到 (B, H, 1, 1) 的外积结果（被后面的 vv @ kk 替代）
        kk = k[:, t, :].view(B, H, 1, N)

        # receptance 向量： (B, H, N) → (B, H, N, 1)
        # 含义：形状变为 "N 行 1 列" 的列向量
        # 用于 output = state @ rr，即用 r 查询状态矩阵
        rr = r[:, t, :].view(B, H, N, 1)

        # value 向量： (B, H, N) → (B, H, N, 1)
        # 含义：列向量，与 kk 做外积 vv @ kk → (B, H, N, N)
        # 将当前 token 的键值关联写入状态矩阵
        vv = v[:, t, :].view(B, H, N, 1)

        # Delta Rule a 向量： (B, H, N) → (B, H, N, 1)
        # 含义：列向量，代表 "要移除的方向"
        # 与 bb 组合形成秩-1 移除矩阵 aa @ bb
        aa = a[:, t, :].view(B, H, N, 1)

        # Delta Rule b 向量： (B, H, N) → (B, H, 1, N)
        # 含义：行向量，代表 "从哪些维度移除"
        # 与 aa 组合形成秩-1 移除矩阵 aa @ bb ∈ R^{N×N}
        bb = b[:, t, :].view(B, H, 1, N)

        # ========== 步骤 2：Delta Rule 状态更新 ==========
        # 公式: state = state * decay + state @ a @ b + v @ k
        # 这是 RWKV-7 的核心创新所在
        #
        # 逐项解释:
        #   (A) state * decay:  指数衰减旧状态（通用遗忘机制）
        #   (B) state @ a @ b:  Delta Rule（选择性/定向移除）
        #   (C) v @ k:          写入新的键值关联（学习新信息）
        #
        # 计算图:
        #   state (B,H,N,N) × decay (B,H,1,N) → (B,H,N,N)  [逐通道衰减]
        #   state (B,H,N,N) @ aa (B,H,N,1) → (B,H,N,1)
        #          @ bb (B,H,1,N)          → (B,H,N,N)     [Delta Rule 移除]
        #   vv (B,H,N,1) @ kk (B,H,1,N)   → (B,H,N,N)     [新键值关联]
        state = (
            # (A) 指数衰减：
            # w[:, t, :] 形状为 (B, H, N)
            # 通过 None 增加维度 → (B, H, 1, N)
            # 广播后与 state (B, H, N, N) 逐元素相乘
            # 效果：state 的第 i 行乘以 w[:, t, :, i]（每个通道独立的衰减因子）
            state * w[:, t, :, None, :]

            # (B) Delta Rule 选择性移除：
            # state @ aa: 先用 a 向量从状态中提取需要移除的内容
            #   state (B,H,N,N) @ aa (B,H,N,1) → (B,H,N,1)
            #   结果表示：对于每个值维度，当前 "a 方向" 上存储了多少信息
            # (state @ aa) @ bb: 再用 b 向量将提取出的信息从状态中按比例抹除
            #   (B,H,N,1) @ bb (B,H,1,N) → (B,H,N,N)
            #   结果是一个秩-1 矩阵，表示要"减去"的内容
            # 由于前面是 += 运算，所以这是将 aa @ bb 的结果加到状态上
            # 注意：a 向量可正可负，因此这既可以是移除也可以是增强
            + state @ aa @ bb

            # (C) 新键值关联：
            # vv (B,H,N,1) @ kk (B,H,1,N) → (B,H,N,N)
            # 外积运算：将 value 向量与 key 向量做外积
            # 结果矩阵中，位置 (i, j) = vv[i] * kk[j]
            # 含义：将当前 token 的 "值" 信息按 key 的分布写入状态
            # 这等价于 Hebbian 学习规则：同时激活的神经元之间的连接被加强
            + vv @ kk
        )

        # ========== 步骤 3：读取输出 ==========
        # 用 receptance 向量 r 查询更新后的状态矩阵
        # state (B,H,N,N) @ rr (B,H,N,1) → (B,H,N,1) → view → (B,H,N)
        #
        # 直观理解：
        #   state 的每一行代表了某维度上的键值关联分布式表示
        #   r 向量是一个门控查询，决定了哪些维度的信息应该流出
        #   state @ r 计算的是：对于每个输出维度，状态中各项的加权和
        #
        # 这与 Transformer 中的 output projection 类似，
        # 但这里的 "注意力分数" 已经是 O(1) 的状态查询，而非 O(n) 的序列匹配
        out[:, t, :] = (state @ rr).view(B, H, N)

    # 返回所有时间步的输出
    # 形状 (B, T, H, N)，每个位置包含该时间步的注意力输出
    return out
