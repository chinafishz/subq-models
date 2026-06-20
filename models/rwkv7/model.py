"""
RWKV-7 "Goose" 语言模型（纯 PyTorch 实现，兼容 MPS 后端）
=============================================================
改编自 BlinkDL/RWKV-LM 仓库的 RWKV-v7/rwkv_v7_demo.py 文件

与原版官方代码的主要区别：
1. 无 CUDA 内核依赖 — 使用纯 PyTorch 实现的 wkv7_operator 算子
2. 不使用 torch.jit.script — 采用普通 nn.Module 以确保 MPS（Apple Metal）兼容性
3. 简化了 value residual（值残差）机制
4. 默认使用 FP32 精度（MPS 后端对 BF16 支持有限）

许可证：Apache 2.0（继承自 RWKV-LM 项目）
"""

import math  # 导入 Python 标准数学库，用于 sqrt 等纯 Python 运算
import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，提供 LayerNorm、Linear 等基础组件
from torch.nn import functional as F  # 导入 PyTorch 的函数式 API，提供 softmax、softplus 等激活函数

from .wkv7_operator import wkv7_forward  # 从同目录下的 wkv7_operator 模块导入 WKV-7 核心算子（纯 PyTorch 实现，无需 CUDA 内核）


class RWKV7Config:
    """配置容器类，用于统一管理 RWKV-7 模型的所有超参数，模仿原版 args 命名空间的接口。"""
    def __init__(self, config_dict: dict):
        # ---- 基础超参数 ----
        self.n_embd = config_dict["n_embd"]  # 模型的主隐藏维度/嵌入维度（embedding dimension），例如小模型常用 512
        self.n_layer = config_dict["n_layer"]  # RWKV-7 Block 的堆叠层数，决定模型深度
        self.vocab_size = config_dict["vocab_size"]  # 词表大小，决定 embedding 矩阵和输出 head 的维度
        self.ctx_len = config_dict.get("ctx_len", 4096)  # 最大上下文长度（context length），训练时的最大序列长度，默认 4096

        # ---- LoRA 降维参数（低秩适配，用于减少参数量） ----
        # RWKV-7 通过低秩分解（LoRA）将多个路径的全连接映射拆分为 输入→低秩→输出 的两步投影
        self.head_size_a = config_dict.get("head_size_a", 64)  # 每个注意力头的维度大小，默认 64
        self.D_DECAY_LORA = config_dict.get("D_DECAY_LORA", 32)  # 衰减路径（decay/w）的 LoRA 中间维度，默认 32
        self.D_AAA_LORA = config_dict.get("D_AAA_LORA", 32)  # 上下文学习率路径（a/aaa）的 LoRA 中间维度，默认 32
        self.D_MV_LORA = config_dict.get("D_MV_LORA", 16)  # 值残差路径（v/mv）的 LoRA 中间维度，默认 16
        self.D_GATE_LORA = config_dict.get("D_GATE_LORA", 64)  # 门控路径（gate/g）的 LoRA 中间维度，默认 64

        # ---- 派生参数（由基础参数计算得出） ----
        self.dim_att = self.n_embd  # 注意力维度等于主隐藏维度（RWKV 的特色：不拆分 QKV，而是同一维度做时间混合）
        self.dim_ffn = int(self.n_embd * 3.5)  # 前馈网络（Channel Mixing）的内部维度，设为 3.5 倍而非传统 4 倍，为 25M 参数量预算做优化
        self.head_size = self.head_size_a  # 每个头的维度（RWKV-7 采用"头维度 = 注意力维度 / 头数"的经典多头设计）
        self.n_head = self.dim_att // self.head_size  # 注意力头数，由主维度 ÷ 每头维度计算
        assert self.dim_att % self.n_head == 0, (  # 断言：主维度必须能被头数整除，否则多头拆分时维度不对齐会导致 reshape 失败
            f"dim_att ({self.dim_att}) must be divisible by n_head ({self.n_head})"  # 断言失败时的错误提示信息
        )


class RWKV_Tmix_x070(nn.Module):
    """RWKV-7 时间混合块（Time Mixing block），是传统 Transformer Attention 的 O(n) 线性复杂度替代方案。

    核心思想：
    1. 用 WKV（Weighted-Key-Value）线性注意力替代 Softmax 自注意力
    2. 引入可学习的衰减率（decay）和上下文学习率（in-context learning rate a）
    3. 通过 Delta Rule（增量规则）实现类似梯度下降的在线学习效果
    4. 使用 token shift（时间移位）让当前 token 能访问前一 token 的信息
    """

    def __init__(self, args: RWKV7Config, layer_id: int):
        super().__init__()  # 调用 nn.Module 的构造函数，注册参数和子模块
        self.args = args  # 保存模型配置引用，方便在 forward 中访问超参数
        self.layer_id = layer_id  # 当前层的编号（从 0 开始），用于特殊处理第 0 层（跳过 value residual）
        self.head_size = args.head_size  # 每头维度 N（如 64），局部变量缓存以提高可读性
        self.n_head = args.n_head  # 头数 H（如 8），局部变量缓存
        C = args.n_embd  # 主隐藏维度 C（如 512），代码中大量使用，简写为 C
        H = self.n_head  # 头数 H，代码中大量使用，简写为 H
        N = self.head_size  # 每头维度 N，代码中大量使用，简写为 N

        # ---- 时间移位参数（Time-shift parameters） ----
        # RWKV 不使用位置编码，而是通过将当前 token 与前一 token 的线性组合来注入时序信息
        # 每个时间移位参数形状为 (1, 1, C)，在广播后与每个 (B, T, C) 的输入相乘
        # 初始化为全 1，即训练开始时不做任何移位（当前 token * 1 + 前一 token * 0）
        self.x_r = nn.Parameter(torch.ones(1, 1, C))  # 接受度（receptance/r）路径的时间移位系数
        self.x_w = nn.Parameter(torch.ones(1, 1, C))  # 衰减（decay/w）路径的时间移位系数
        self.x_k = nn.Parameter(torch.ones(1, 1, C))  # 键（key/k）路径的时间移位系数
        self.x_v = nn.Parameter(torch.ones(1, 1, C))  # 值（value/v）路径的时间移位系数
        self.x_a = nn.Parameter(torch.ones(1, 1, C))  # 上下文学习率（aaa/a）路径的时间移位系数
        self.x_g = nn.Parameter(torch.ones(1, 1, C))  # 门控（gate/g）路径的时间移位系数

        # ---- 衰减路径（Decay path, w）：低秩投影 ----
        # 衰减率 w 控制模型对历史信息的遗忘速度
        # 通过 LoRA 低秩分解实现：w = w0 + tanh(xw @ w1 @ w2)，再通过 softplus 映射到负数区间
        D_DECAY = args.D_DECAY_LORA  # LoRA 中间维度（默认 32），远小于 C（如 512）
        self.w0 = nn.Parameter(torch.zeros(1, 1, C))  # 衰减的基准偏置（per-channel bias），初始化为零
        self.w1 = nn.Parameter(torch.zeros(C, D_DECAY))  # 衰减 LoRA 的下投影矩阵（压缩）：C → D_DECAY
        self.w2 = nn.Parameter(torch.zeros(D_DECAY, C))  # 衰减 LoRA 的上投影矩阵（还原）：D_DECAY → C

        # ---- 上下文学习率路径（In-context learning rate path, a） ----
        # 参数 a（有时叫 aaa）控制模型对新输入信息的信任程度，类似于梯度下降中的学习率
        # 通过 LoRA 低秩分解实现：a = sigmoid(a0 + xa @ a1 @ a2)
        D_AAA = args.D_AAA_LORA  # LoRA 中间维度（默认 32）
        self.a0 = nn.Parameter(torch.zeros(1, 1, C))  # 学习率的基准偏置，初始化为零
        self.a1 = nn.Parameter(torch.zeros(C, D_AAA))  # 学习率 LoRA 的下投影矩阵：C → D_AAA
        self.a2 = nn.Parameter(torch.zeros(D_AAA, C))  # 学习率 LoRA 的上投影矩阵：D_AAA → C

        # ---- 值残差路径（Value residual path, v） ----
        # 值残差为 Value 向量添加一个可学习的残差修正项
        # 第 0 层不创建值残差参数（与官方 RWKV-7 行为一致），因为第 0 层没有前一层可借鉴的残差
        D_MV = args.D_MV_LORA  # LoRA 中间维度（默认 16）
        self.v0 = nn.Parameter(torch.zeros(1, 1, C)) if layer_id > 0 else None  # 值残差基准偏置，仅非第 0 层创建
        self.v1 = nn.Parameter(torch.zeros(C, D_MV)) if layer_id > 0 else None  # 值残差 LoRA 下投影，仅非第 0 层创建
        self.v2 = nn.Parameter(torch.zeros(D_MV, C)) if layer_id > 0 else None  # 值残差 LoRA 上投影，仅非第 0 层创建

        # ---- 门控路径（Gate path, g） ----
        # 门控信号 g 用于调制输出，决定最终的输出强度
        # 通过 LoRA 低秩分解实现：g = sigmoid(xg @ g1) @ g2
        D_GATE = args.D_GATE_LORA  # LoRA 中间维度（默认 64）
        self.g1 = nn.Parameter(torch.zeros(C, D_GATE))  # 门控 LoRA 的下投影矩阵：C → D_GATE
        self.g2 = nn.Parameter(torch.zeros(D_GATE, C))  # 门控 LoRA 的上投影矩阵：D_GATE → C

        # ---- Key 归一化参数 ----
        # RWKV 对 Key 向量进行 L2 归一化（除以模长），使注意力计算更稳定
        self.k_k = nn.Parameter(torch.ones(1, 1, C))  # Key 强度调节系数（per-channel scaling），初始化为 1
        self.k_a = nn.Parameter(torch.ones(1, 1, C))  # Key 的学习率调节系数，用于配合 a 参数调节 key 方向，初始化为 1
        self.r_k = nn.Parameter(torch.zeros(H, N))  # 局部注意力的 key-receptance 交互核，形状为 (头数, 头维度)

        # ---- 主要线性投影层 ----
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 时间移位操作：用零填充实现序列右移一位，当前token看到前一token信息
        self.receptance = nn.Linear(C, C, bias=False)  # 接受度投影（相当于 Transformer 的 Query），无偏置
        self.key = nn.Linear(C, C, bias=False)  # 键投影（相当于 Transformer 的 Key），无偏置
        self.value = nn.Linear(C, C, bias=False)  # 值投影（相当于 Transformer 的 Value），无偏置
        self.output = nn.Linear(C, C, bias=False)  # 输出投影（将时间混合结果映射回隐藏空间），无偏置
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)  # 层归一化：将 C 维分为 H 组（每组 N 维），用 GroupNorm 做 per-head 归一化

        # 初始化权重（RWKV 特色的全零初始化策略）
        self._init_weights()

    def _init_weights(self):
        """RWKV 风格的权重初始化：大部分线性层采用零初始化（zero-init）。

        设计原理：
        - 零初始化的层在训练开始时输出为零，相当于"跳过"该路径
        - 随时间移位参数（初始值为 1），模型逐步学习何时启用各条路径
        - 这种策略让训练从简单的 token shift 开始，逐步增加复杂度
        """
        nn.init.zeros_(self.output.weight)  # 输出投影的权重矩阵初始化为全零
        nn.init.zeros_(self.w1)  # 衰减 LoRA 下投影初始化为全零
        nn.init.zeros_(self.w2)  # 衰减 LoRA 上投影初始化为全零
        nn.init.zeros_(self.a1)  # 学习率 LoRA 下投影初始化为全零
        nn.init.zeros_(self.a2)  # 学习率 LoRA 上投影初始化为全零
        if self.v1 is not None:  # 如果是非第 0 层（v1 不为 None）
            nn.init.zeros_(self.v1)  # 值残差 LoRA 下投影初始化为全零
            nn.init.zeros_(self.v2)  # 值残差 LoRA 上投影初始化为全零
        nn.init.zeros_(self.g1)  # 门控 LoRA 下投影初始化为全零
        nn.init.zeros_(self.g2)  # 门控 LoRA 上投影初始化为全零

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RWKV-7 时间混合前向传播。

        算法流程概述：
        1. 时间移位：让当前 token 能混合前一 token 的信息
        2. 计算六个路径：r（接受度/query）、w（衰减率）、k（键）、v（值）、a（学习率）、g（门控）
        3. WKV Delta Rule 核心计算：线性注意力替代 softmax 自注意力
        4. 局部注意力加成
        5. 输出门控

        参数：
            x: 输入张量，形状为 (B, T, C)
               B = batch size（批次大小）
               T = sequence length（序列长度）
               C = embedding dimension（隐藏维度）

        返回：
            时间混合后的输出张量，形状为 (B, T, C)
        """
        B, T, C = x.shape  # 解包输入形状：批次大小 B、序列长度 T、隐藏维度 C
        H, N = self.n_head, self.head_size  # 局部变量缓存头数 H 和每头维度 N

        # ==================== 步骤 1：时间移位（Token Shift） ====================
        # RWKV 的核心创新：不使用位置编码，而是让当前 token 直接访问前一 token 的隐藏状态
        # time_shift = ZeroPad2d((0,0, 1,-1)) 将序列右移一位，首位补零
        # xx = shifted_x - x 得到"前一 token 与当前 token 的差异向量"
        xx = self.time_shift(x) - x  # xx 形状: (B,T,C)，每个位置是前一个 token 与当前 token 的差值

        # 各路径分别进行时间移位混合：x_path = x + xx * x_shift_coef
        # x_shift_coef 是可学习的 per-channel 系数，控制每个维度混合前一个 token 信息的程度
        xr = x + xx * self.x_r  # 接受度路径的时间混合输入，供后续计算 query 使用
        xw = x + xx * self.x_w  # 衰减路径的时间混合输入，供后续计算 decay rate 使用
        xk = x + xx * self.x_k  # 键路径的时间混合输入，供后续计算 key 使用
        xv = x + xx * self.x_v  # 值路径的时间混合输入，供后续计算 value 使用
        xa = x + xx * self.x_a  # 学习率路径的时间混合输入，供后续计算 in-context learning rate 使用
        xg = x + xx * self.x_g  # 门控路径的时间混合输入，供后续计算 gate 使用

        # ==================== 步骤 2：接受度（Receptance / Query） ====================
        # 接受度 r 相当于 Transformer 中的 Query，决定模型"要读取什么信息"
        r = self.receptance(xr)  # 线性投影：xr → r，形状 (B, T, C)

        # ==================== 步骤 3：衰减率（Decay Rate / w） ====================
        # w 控制模型对历史信息的遗忘速度
        # 计算公式：w = -softplus(-(w0 + tanh(xw @ w1 @ w2))) - 0.5
        #   xw @ w1 @ w2：通过 LoRA 低秩分解学习输入相关的衰减值
        #   tanh：将 LoRA 输出限制在 (-1, 1)，增强训练稳定性
        #   w0 + ...：加上 per-channel 可学习偏置
        #   -softplus(-x) = -log(1+exp(-x))：soft-clamp 到负区间，确保 w ≤ 0
        #   -0.5：将 w 进一步限制在 (-inf, -0.5)，即衰减率永远小于 -0.5
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5  # 形状 (B, T, C)

        # ==================== 步骤 4：键和值（Key & Value） ====================
        k = self.key(xk)  # 键投影：xk → k，形状 (B, T, C)，相当于 Transformer 的 Key
        v = self.value(xv)  # 值投影：xv → v，形状 (B, T, C)，相当于 Transformer 的 Value

        # ==================== 步骤 5：值残差（Value Residual） ====================
        # 第 0 层跳过（layer_id == 0 或 v0 为 None），与官方 RWKV-7 行为一致
        # 值残差：v = v + sigmoid(v0 + (xv @ v1 @ v2))
        #   xv @ v1 @ v2：通过 LoRA 低秩分解学习输入相关的值修正
        #   v0：per-channel 偏置
        #   sigmoid：将残差限制在 (0, 1)，提供温和的修正幅度
        if self.layer_id > 0 and self.v0 is not None:  # 仅在非第 0 层执行值残差
            v = v + (torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2))  # 形状 (B, T, C)

        # ==================== 步骤 6：上下文学习率（In-context Learning Rate / a） ====================
        # a 决定了模型对新输入信息的信任程度，类似梯度下降中的学习率
        # 计算公式：a = sigmoid(a0 + (xa @ a1 @ a2))
        #   xa @ a1 @ a2：通过 LoRA 低秩分解学习输入相关的学习率
        #   a0：per-channel 偏置
        #   sigmoid：将 a 限制在 (0, 1)，即学习率在 0 到 1 之间
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)  # 形状 (B, T, C)

        # ==================== 步骤 7：门控（Gate / g） ====================
        # g 控制最终输出的调制强度
        # 计算公式：g = sigmoid(xg @ g1) @ g2
        #   xg @ g1 @ g2：通过 LoRA 低秩分解学习输入相关的门控信号
        #   sigmoid：将门控限制在 (0, 1)，控制信息流通量
        g = torch.sigmoid(xg @ self.g1) @ self.g2  # 形状 (B, T, C)

        # ==================== 步骤 8：Key 归一化和调制 ====================
        # 计算 kk：用 k_k 缩放后的 key
        kk = k * self.k_k  # per-channel 缩放，形状 (B, T, C)
        # L2 归一化：将 kk 按头维度归一化为单位向量
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)  # reshape → 归一化 → reshape 回

        # 调制原始的 k：k = k * (1 + (a - 1) * k_a)
        #   当 a=1 时，k 不变；当 a=0 时，k = k * (1 - k_a)
        #   这使学习率 a 能动态调节 key 的幅度，a 越大 key 越强
        k = k * (1 + (a - 1) * self.k_a)  # 形状 (B, T, C)

        # ==================== 步骤 9：WKV Delta Rule 核心计算 ====================
        # 这是 RWKV-7 的核心创新：将线性注意力写为增量规则（Delta Rule）的形式
        #
        # 传统线性注意力：output = (Q @ K^T) @ V
        # WKV Delta Rule： state[t] = state[t-1] * decay + delta_update
        #
        # 衰减率计算：w_decay = exp(-exp(w))
        #   - 内层 exp(w)：将 w（≤-0.5）映射到 (0, exp(-0.5)) ≈ (0, 0.607)
        #   - 外层 exp(-...)：再取负指数，得到在 (0.545, 1) 范围的衰减因子
        #   - 值越接近 0 表示遗忘越快，越接近 1 表示记忆越持久
        #   - 转换为 float() 是因为 MPS 后端对 bf16 支持有限，强制使用 FP32
        w_decay = torch.exp(-torch.exp(w.view(B, T, H, N).float()))  # 形状 (B, T, H, N)

        # wkv7_forward 参数说明：
        #   r: 接受度（query），形状 (B,T,H,N) — 控制"读取"什么
        #   w: 逐 token 的衰减因子，形状 (B,T,H,N) — 控制"遗忘"速度
        #   k: 键（key），形状 (B,T,H,N) — 被查询的内容
        #   v: 值（value），形状 (B,T,H,N) — 实际取出的信息
        #   -kk: 负的归一化 key，形状 (B,T,H,N) — 用作 Delta Rule 中的"移除项"
        #        Delta Rule 更新公式：state = state * decay + v * k^T - v_old * k_old^T
        #        -kk 就是公式中的 -k_old^T，用于从状态中移除"旧键对应值"的贡献
        #   kk*a: 调制后的辅助 key，形状 (B,T,H,N) — 用作 Delta Rule 的"辅助添加项"
        #        kk*a 在输出中提供受学习率 a 调制的额外信息通道
        x = wkv7_forward(  # 调用纯 PyTorch 实现的 WKV-7 核心算子
            r.view(B, T, H, N).float(),  # 将接受度 reshape 为多头格式并转为 FP32
            w_decay,  # 衰减因子矩阵
            k.view(B, T, H, N).float(),  # 将 key reshape 为多头格式并转为 FP32
            v.view(B, T, H, N).float(),  # 将 value reshape 为多头格式并转为 FP32
            (-kk).view(B, T, H, N).float(),         # -kk：移除项（不是 a！），Delta Rule 中用于"忘记旧信息"
            (kk * a).view(B, T, H, N).float(),      # kk * a：调制辅助项，学习率 a 控制其强度
        ).view(B, T, C).to(x.dtype)  # reshape 回 (B,T,C) 并转回输入的原始精度（可能是 FP32 或 FP16）

        # ==================== 步骤 10：层归一化 ====================
        # 使用 GroupNorm（按头分组）而非 LayerNorm，因为 GroupNorm 在 MPS 上性能更好
        # 先将 B*T 合并为一批，做完归一化后再恢复形状
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        # ==================== 步骤 11：局部注意力加成（Local Attention Bonus） ====================
        # 除了 WKV 的全局线性注意力外，RWKV-7 还添加了一项"局部注意力 bonus"
        # 计算方式：对于每个头，计算 r 和 k 的点积（element-wise），通过 r_k 做 per-head 加权利
        # 然后与 v 逐元素相乘后求和
        # 公式：(r * k * r_k).sum(dim=-1) * v
        #   这相当于在每个头的维度上计算"当前 token 的 r 和 k 的匹配度"，用该标量加权 v
        x = x + (  # 将局部注意力加成累加到主输出上
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)  # 多头格式下的 r * k * r_k，形状 (B,T,H,N)
            .sum(dim=-1, keepdim=True)  # 沿头维度求和得到标量，keepdim=True 保持形状以便广播，结果 (B,T,H,1)
            * v.view(B, T, H, -1)  # 与多头 value 逐元素相乘，广播到 (B,T,H,N)
        ).view(B, T, C)  # reshape 回 (B, T, C)

        # ==================== 步骤 12：输出门控 ====================
        # 用 gate 调制输出：output = Linear(x * g)
        #   类似于 LSTM/GRU 的 output gate，控制有多少信息从时间混合块流出
        x = self.output(x * g)  # 先做 element-wise 门控乘法，再线性投影到输出空间，形状 (B, T, C)
        return x  # 返回时间混合块输出


class RWKV_CMix_x070(nn.Module):
    """RWKV-7 通道混合块（Channel Mixing block），是传统 Transformer FFN 的替代方案。

    与标准 FFN（两个线性层 + 激活函数）的区别：
    1. 使用 ReLU²（ReLU 平方）激活函数，比 GELU 更简单且在小模型中效果相当
    2. 包含时间移位操作，让 FFN 也能利用前一 token 的信息
    3. 将隐藏维度从 C 扩展到 dim_ffn（3.5C）再压缩回 C
    """

    def __init__(self, args: RWKV7Config, layer_id: int):
        super().__init__()  # 调用 nn.Module 构造函数
        self.args = args  # 保存配置引用
        self.layer_id = layer_id  # 层编号（当前未使用，保留以备后续扩展）
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 时间移位操作：序列右移一位，首位补零
        self.x_k = nn.Parameter(torch.ones(1, 1, args.n_embd))  # 时间移位系数，初始化为 1（不做移位）
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)  # 上投影：C → dim_ffn（3.5C），无偏置
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)  # 下投影：dim_ffn → C，无偏置

        # 值投影权重零初始化（RWKV 风格：训练开始时通道混合块输出为零）
        nn.init.zeros_(self.value.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        通道混合前向传播。

        算法流程：
        1. 时间移位：混合前一 token 的隐藏状态
        2. 上投影 + ReLU² 激活
        3. 下投影还原

        参数：
            x: 输入张量，形状 (B, T, C)

        返回：
            通道混合后的输出，形状 (B, T, C)
        """
        xx = self.time_shift(x) - x  # 时间移位差值，形状 (B,T,C)：每个位置是前一token与当前token的差
        k = x + xx * self.x_k  # 时间混合：当前 token + 可学习系数 * 前一 token 差值
        k = torch.relu(self.key(k)) ** 2  # ReLU² 激活：上投影 → ReLU 截断负值 → 平方（增强稀疏性和非线性）
        return self.value(k)  # 下投影还原到隐藏维度 C，形状 (B, T, C)


class Block(nn.Module):
    """RWKV-7 基础块：由时间混合（TimeMix）和通道混合（ChannelMix）组成，带有残差连接。

    架构（Pre-Norm 风格，与大多数现代 Transformer 一致）：
        x = x + TimeMix(LayerNorm(x))   # 时间混合 + 残差连接
        x = x + ChannelMix(LayerNorm(x))  # 通道混合 + 残差连接
    """

    def __init__(self, args: RWKV7Config, layer_id: int):
        super().__init__()  # 调用 nn.Module 构造函数
        self.args = args  # 保存配置引用
        self.layer_id = layer_id  # 层编号

        self.ln1 = nn.LayerNorm(args.n_embd)  # 时间混合前的层归一化（Pre-Norm），稳定训练
        self.ln2 = nn.LayerNorm(args.n_embd)  # 通道混合前的层归一化（Pre-Norm），稳定训练
        self.att = RWKV_Tmix_x070(args, layer_id)  # 时间混合子模块（WKV 注意力替代方案）
        self.ffn = RWKV_CMix_x070(args, layer_id)  # 通道混合子模块（FFN 替代方案）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：Pre-Norm → 子模块 → 残差连接。

        参数：
            x: 输入张量，形状 (B, T, C)

        返回：
            Block 输出张量，形状 (B, T, C)
        """
        x = x + self.att(self.ln1(x))  # 第一子层：LayerNorm → 时间混合（WKV attention）→ 残差相加
        x = x + self.ffn(self.ln2(x))  # 第二子层：LayerNorm → 通道混合（ReLU² FFN）→ 残差相加
        return x  # 返回 Block 输出


class CourageLM(nn.Module):
    """
    RWKV-7 Courage 语言模型。

    一个约 25M 参数的语言模型，预训练数据中融入了数码宝贝"勇气徽章"精神。

    整体架构：
        Embedding → [Block × n_layer] → LayerNorm → Head

    每个 Block 的结构：
        LayerNorm → TimeMix（WKV 注意力）→ +残差连接
        LayerNorm → ChannelMix（ReLU² FFN）→ +残差连接

    特点：
    - 输入/输出 embedding 权重绑定（weight tying），节省参数
    - 第 0 层前有一个额外的 LayerNorm（RWKV 惯例）
    """

    def __init__(self, config: RWKV7Config):
        super().__init__()  # 调用 nn.Module 构造函数
        self.config = config  # 保存模型配置对象

        # ---- Embedding 层 ----
        # 将 token ID（整数）映射为稠密向量
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)  # 词嵌入矩阵，形状 (vocab_size, n_embd)

        # ---- 第 0 层前置归一化（RWKV 惯例） ----
        # 在进入第一个 Block 之前，对嵌入向量做一次 LayerNorm
        # 这有助于稳定训练初期的梯度传播
        self.ln0 = nn.LayerNorm(config.n_embd)  # 前置 LayerNorm，对 embedding 输出做归一化

        # ---- RWKV-7 堆叠块 ----
        # 创建 n_layer 个 RWKV-7 Block，每个 Block 包含时间混合和通道混合
        self.blocks = nn.ModuleList([  # 用 ModuleList 包装，确保 PyTorch 能正确追踪参数
            Block(config, i) for i in range(config.n_layer)  # 逐层创建 Block，层编号从 0 到 n_layer-1
        ])

        # ---- 输出层 ----
        self.ln_out = nn.LayerNorm(config.n_embd)  # 输出前的最后一次归一化
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 输出头：将隐藏状态映射到词表 logits，无偏置

        # ---- 权重绑定（Weight Tying） ----
        # 输入 embedding 和输出 head 共享权重矩阵
        # 这在小型语言模型中很常见，可减少约 vocab_size * n_embd 个参数
        self.head.weight = self.emb.weight  # 共享权重：head 和 emb 指向同一个 Parameter 对象

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        模型前向传播。

        参数：
            idx: token 索引张量，形状为 (B, T)
                 B = batch size（批次大小）
                 T = sequence length（序列长度）

        返回：
            logits 张量，形状为 (B, T, vocab_size)
            每个位置对应词汇表中每个 token 的未归一化对数概率
        """
        B, T = idx.shape  # 解包输入形状：批次大小 B、序列长度 T

        # ---- 嵌入 + 第 0 层前置归一化 ----
        x = self.ln0(self.emb(idx))  # token ID → embedding 向量 → LayerNorm，形状 (B, T, n_embd)

        # ---- 逐层通过 RWKV-7 Block 堆叠 ----
        for block in self.blocks:  # 遍历所有 Block（从第 0 层到第 n_layer-1 层）
            x = block(x)  # 通过当前 Block：Pre-Norm → TimeMix →残差 → Pre-Norm → ChannelMix → 残差

        # ---- 输出投影 ----
        x = self.ln_out(x)  # 最后的 LayerNorm 归一化，形状 (B, T, n_embd)
        logits = self.head(x)  # 线性投影到词表空间，形状 (B, T, vocab_size)

        return logits  # 返回未归一化的对数概率（可用于计算交叉熵损失或 argmax 生成）

    def count_parameters(self) -> int:
        """统计模型中所有可训练参数的数量（不包含冻结参数）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # 累加所有 requires_grad=True 的参数的元素个数

    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        自回归文本生成。

        参数：
            idx: 起始 token 索引张量，形状 (B, T)，作为生成的前缀/提示词
            max_new_tokens: 要生成的新 token 数量
            temperature: 采样温度参数（默认 1.0 表示不做温度调节）
                        温度 > 1：增加随机性/多样性
                        温度 < 1：减少随机性，更确定

        返回：
            生成的完整序列，形状 (B, T + max_new_tokens)
            包含原始前缀和所有新生成的 token
        """
        for _ in range(max_new_tokens):  # 循环生成 max_new_tokens 个 token
            # ---- 裁剪到上下文长度 ----
            # 只保留最近 ctx_len 个 token 作为输入（防止超出模型最大上下文）
            idx_cond = idx[:, -self.config.ctx_len:]  # 沿序列维度取最后 ctx_len 个位置

            # ---- 前向传播（推理模式，不计算梯度） ----
            with torch.no_grad():  # 禁用梯度计算以节省显存和加速推理
                logits = self(idx_cond)  # 前向传播得到 logits，形状 (B, cond_len, vocab_size)

            # ---- 只取最后一个时间步的 logits ----
            # 因为是自回归生成，我们只需要最后一个位置的预测来做下一个 token 的采样
            logits = logits[:, -1, :] / temperature  # 取最后时间步 → 除以温度调节分布尖锐度

            # ---- 采样下一个 token ----
            probs = F.softmax(logits, dim=-1)  # 将温度调节后的 logits 转为概率分布
            idx_next = torch.multinomial(probs, num_samples=1)  # 按概率采样 1 个 token（多项式采样）

            # ---- 将新 token 拼接到序列末尾 ----
            idx = torch.cat((idx, idx_next), dim=1)  # 沿序列维度拼接新 token，序列长度 +1
        return idx  # 返回完整序列（原始前缀 + 所有生成的 token）
