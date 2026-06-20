"""
RWKV-Courage 训练循环 (Training Loop)
============================
RWKV-7 "Goose" 语言模型的主训练脚本。
兼容 MPS (Apple Silicon)，无需 CUDA。

使用方法 (Usage):
    python -m src.train --config configs/courage_25m.yaml
"""

import argparse  # 命令行参数解析库，用于从命令行接收配置文件和断点续训路径
import os  # 操作系统接口，用于文件路径操作、目录创建、文件存在性检查
import sys  # 系统相关功能，用于修改模块搜索路径和程序退出
import time  # 时间相关功能，用于计时和计算训练速度（tokens/秒）
import math  # 数学函数库，用于余弦学习率调度中的 cos() 和 pi 计算
from typing import Optional  # 类型注解，Optional[X] 表示值是 X 类型或 None
import yaml  # YAML 解析库，用于读取训练配置文件（config.yaml）
import torch  # PyTorch 深度学习框架主模块
import torch.nn.functional as F  # PyTorch 函数式接口，提供 cross_entropy 等损失函数
from pathlib import Path  # 面向对象的文件路径操作类，比 os.path 更现代化

# 将父目录添加到 Python 模块搜索路径，以便导入 src 包中的模块
# __file__ 是当前文件路径，.parent.parent 向上两级得到项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CourageLM, RWKV7Config  # 导入 RWKV-7 模型类和配置类
from src.dataset import MixedDataset  # 导入混合数据集类（TinyStories + Courage 素材的混合数据流）
from tokenizers import Tokenizer  # 导入 HuggingFace tokenizers 库的 Tokenizer 类


def get_device(config: dict) -> torch.device:
    """自动检测最佳可用设备：MPS > CUDA > CPU (Auto-detect best available device)"""
    # 从配置字典中读取用户指定的设备偏好，默认为 "auto"（自动选择）
    # config.get("training", {}) —— 如果 "training" 键不存在，返回空字典避免 KeyError
    requested = config.get("training", {}).get("device", "auto")

    # 情况1：用户明确要求 MPS，且 MPS 后端可用（Apple Silicon GPU）
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")  # 返回 MPS 设备（Metal Performance Shaders）
    # 情况2：用户明确要求 CUDA，且 NVIDIA GPU 可用
    elif requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")  # 返回 CUDA 设备
    # 情况3：用户设为 "auto" 或任意非 CPU 的值，自动选择最佳设备
    elif requested != "cpu":
        if torch.backends.mps.is_available():
            return torch.device("mps")  # 优先 MPS（Apple Silicon 最常用）
        elif torch.cuda.is_available():
            return torch.device("cuda")  # 其次 CUDA

    return torch.device("cpu")  # 兜底：CPU 训练（最慢但兼容性最好）


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件 (Load YAML configuration)"""
    # 以只读模式打开配置文件
    with open(config_path, "r") as f:
        # yaml.safe_load() 安全解析 YAML（不会执行任意 Python 代码，比 yaml.load() 安全）
        return yaml.safe_load(f)


def save_checkpoint(
    model: CourageLM,              # 要保存的 RWKV-7 模型实例
    optimizer: torch.optim.Optimizer,  # 要保存的优化器实例（包含动量等状态）
    step: int,                    # 当前训练步数（优化器更新次数）
    loss: float,                  # 当前的平均损失值
    output_dir: str,              # 检查点输出目录
):
    """保存模型检查点 (Save model checkpoint)"""
    # 创建输出目录（exist_ok=True 表示目录已存在时不报错）
    os.makedirs(output_dir, exist_ok=True)
    # 生成检查点文件名，如 checkpoint_001000.pt（6位零填充步数）
    path = os.path.join(output_dir, f"checkpoint_{step:06d}.pt")

    # 构建检查点字典，包含恢复训练所需的所有状态
    checkpoint = {
        "step": step,                              # 当前步数，恢复时知道从哪里继续
        "loss": loss,                              # 当前损失，用于追踪训练进度
        "model_state_dict": model.state_dict(),    # 模型参数（权重和偏置）
        "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态（AdamW 动量、方差等）
        "config": model.config.__dict__,           # 模型配置（词汇表大小、层数、隐藏维度等）
    }
    # 使用 torch.save 将检查点序列化保存为 .pt 文件
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")  # 打印保存路径以确认


def load_checkpoint(
    path: str,                                        # 检查点文件路径
    model: CourageLM,                                 # 要加载参数的模型
    optimizer: Optional[torch.optim.Optimizer] = None,  # 可选：要加载状态的优化器
) -> int:
    """加载模型检查点，返回步数 (Load model checkpoint. Returns the step number)"""
    # 从磁盘加载检查点，map_location="cpu" 确保即使 GPU 训练也能在任意设备上加载
    checkpoint = torch.load(path, map_location="cpu")
    # 将保存的模型参数加载到当前模型实例中（严格匹配参数名）
    model.load_state_dict(checkpoint["model_state_dict"])
    # 如果提供了优化器，也恢复优化器状态（AdamW 的动量、方差累积值）
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # 返回保存的步数，如果检查点中没有 "step" 键则默认返回 0
    return checkpoint.get("step", 0)


def train(config_path: str, resume_from: Optional[str] = None):
    """主训练函数 (Main training function)"""
    # ==================== 加载配置 ====================
    cfg = load_config(config_path)                     # 从 YAML 文件加载完整配置字典
    model_cfg = RWKV7Config(cfg["model"])              # 用 model 部分的配置初始化模型配置对象
    train_cfg = cfg["training"]                        # 提取训练超参数子字典（学习率、批次大小等）

    device = get_device(cfg)                           # 自动选择最佳计算设备
    dtype = torch.float32                              # MPS 上最安全的数据类型（float16 在 MPS 上不稳定）
    print(f"Device: {device} | MPS available: {torch.backends.mps.is_available()}")  # 打印设备信息

    # ==================== 加载分词器 ====================
    # 从配置中获取分词器路径，默认为 data/tokenizer.json
    tokenizer_path = cfg.get("tokenizer_path", "data/tokenizer.json")
    # 检查分词器文件是否存在，若不存在则打印错误信息并退出
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        print("  Run: python -m src.tokenizer_train --input <files> --output data/tokenizer.json")
        sys.exit(1)  # 退出码 1 表示错误退出

    tokenizer = Tokenizer.from_file(tokenizer_path)    # 从文件加载预训练的分词器
    print(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}")  # 打印词汇表大小

    # ==================== 创建模型 ====================
    # 实例化 CourageLM 模型，.to(device) 将模型参数移动到目标设备（MPS/CUDA/CPU）
    # .to(dtype) 将参数转换为指定数据类型（float32）
    model = CourageLM(model_cfg).to(device).to(dtype)
    n_params = model.count_parameters()                # 统计模型总参数量
    print(f"Model: {n_params:,} parameters ({n_params / 1e6:.1f}M)")  # 打印参数量（带千分位分隔符和百万单位）

    # ==================== 创建优化器 ====================
    # AdamW 优化器 —— Adam 的解耦权重衰减变体，是 Transformer 训练的标配
    optimizer = torch.optim.AdamW(
        model.parameters(),                             # 模型所有可训练参数
        lr=train_cfg["learning_rate"],                 # 初始学习率（必需）
        betas=(train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.95)),  # Adam 动量系数
        # beta1=0.9: 一阶矩估计的指数衰减率
        # beta2=0.95: 二阶矩估计的指数衰减率（通常 0.999，这里用 0.95 更快适应）
        weight_decay=train_cfg.get("weight_decay", 0.1),  # 权重衰减（L2 正则化强度），默认 0.1
    )

    # ==================== 数据加载 ====================
    # MixedDataset 混合了两个数据源的比例：
    # - TinyStories: 儿童故事语料（通用语言能力）
    # - Courage Material: 自定义领域语料（特定知识注入）
    dataset = MixedDataset(
        tokenizer=tokenizer,                                        # 分词器实例
        tinystories_ratio=train_cfg["data_mix"]["tinystories"],     # TinyStories 的混合比例
        courage_ratio=train_cfg["data_mix"]["courage_material"],    # Courage 素材的混合比例
        seq_len=model_cfg.ctx_len,                                  # 序列长度（上下文窗口大小）
    )

    # 获取 Courage 素材文件的路径，默认为 data/raw/courage_material.txt
    courage_path = cfg.get("courage_material_path", "data/raw/courage_material.txt")
    # stream_tokens() 返回一个生成器，每次 yield 一个 token ID 序列（张量）
    # 流式处理避免一次性将整个数据集加载到内存
    data_stream = dataset.stream_tokens(courage_path)

    # ==================== 断点续训 ====================
    start_step = 0                                     # 起始步数，默认从 0 开始
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")  # 检查点保存目录
    # 优先使用命令行 --resume 参数，其次使用配置文件中的 resume_from 字段
    resume = resume_from or cfg.get("resume_from")
    if resume and os.path.exists(resume):              # 如果提供了断点路径且文件存在
        start_step = load_checkpoint(resume, model, optimizer)  # 加载检查点，恢复模型和优化器状态
        print(f"Resumed from step {start_step}")       # 打印恢复的步数

    # ==================== 训练超参数提取 ====================
    batch_size = train_cfg["batch_size"]               # 每个微步的批次大小（Micro Batch Size）
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)  # 梯度累积步数，默认 1（不累积）
    # 有效批次大小 = batch_size × grad_accum（在 GPU 内存不足时用小 batch + 多步累积模拟大批次）
    max_steps = train_cfg["max_steps"]                 # 最大训练步数（优化器更新次数）
    warmup_steps = train_cfg["warmup_steps"]           # 学习率预热步数（从 0 线性增长到 base_lr）
    min_lr = train_cfg.get("min_lr", 3e-5)            # 余弦退火的最小学习率，默认 3e-5
    base_lr = train_cfg["learning_rate"]               # 基础学习率（预热结束后的峰值）

    model.train()                                      # 将模型设置为训练模式（启用 dropout 等训练专属行为）
    step = start_step                                  # 当前微步计数器（从断点步数开始）
    total_loss = 0.0                                   # 日志区间内的累积损失（用于计算平均损失）
    last_avg_loss = float("inf")                       # 上一个日志区间的平均损失（检查点元数据用），初始为正无穷
    best_loss = float("inf")                           # 历史最佳损失，初始为正无穷
    lr = base_lr                                       # 当前学习率，初始为 base_lr，随调度动态更新
    t0 = time.time()                                   # 记录日志区间的起始时间

    # opt_step 是优化器更新步数（考虑梯度累积后的实际权重更新次数），与微步 step 不同
    # 例如 grad_accum=4 时，每 4 个微步才产生 1 次优化器更新
    opt_step = start_step
    optimizer.zero_grad()                              # 清空梯度缓存（避免上次训练的梯度残留）

    print(f"\nTraining: batch_size={batch_size}, grad_accum={grad_accum}")
    print(f"  Effective batch = {batch_size * grad_accum}")  # 打印有效批次大小
    print(f"  Max steps = {max_steps} (optimizer updates)\n")

    # 预分配批次张量 —— 复用内存避免每次迭代都重新分配，提升效率
    # x_batch: 输入序列 (batch_size, seq_len)，long 类型因为存的是 token ID
    x_batch = torch.zeros(batch_size, model_cfg.ctx_len, dtype=torch.long, device=device)
    # y_batch: 目标序列 (batch_size, seq_len)，每个位置是 x_batch 对应位置的下一个 token
    y_batch = torch.zeros(batch_size, model_cfg.ctx_len, dtype=torch.long, device=device)

    # ==================== 主训练循环 ====================
    while step < max_steps:
        step += 1                                      # 微步计数器递增

        # --- 获取一批序列数据 ---
        # 对批次中的每个样本分别获取一条序列
        for b in range(batch_size):
            try:
                seq = next(data_stream)                # 从数据流生成器获取下一个 token 序列张量
            except StopIteration:
                # 如果数据流耗尽（遍历完一遍数据集），重新创建数据流（开启新一轮 epoch）
                data_stream = dataset.stream_tokens(courage_path)
                seq = next(data_stream)                # 重新获取一条序列
            # seq[:-1]: 输入序列（去掉最后一个 token）→ 模型输入
            # .to(device) 移动到 GPU/MPS
            x_batch[b] = seq[:-1].to(device)
            # seq[1:]: 目标序列（去掉第一个 token）→ 预测目标（下一个 token 预测，next-token prediction）
            y_batch[b] = seq[1:].to(device)

        # --- 前向传播 ---
        # model(x_batch) 返回 logits: (batch_size, seq_len, vocab_size)
        # 每个位置输出词汇表中每个 token 的未归一化分数
        logits = model(x_batch)
        # 交叉熵损失：比较模型预测与真实下一个 token
        # logits.view(-1, vocab_size): 将所有位置和批次展平为 (B*seq_len, vocab_size)
        # y_batch.view(-1): 展平为 (B*seq_len,) —— 每个位置的真实 token ID
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),          # 输入：展平后的 logits
            y_batch.view(-1),                          # 目标：展平后的真实 token ID
            ignore_index=tokenizer.token_to_id("[PAD]") or -100,  # 忽略填充 token 的损失贡献
            # 如果分词器没有 [PAD] token，退化为 -100（PyTorch 默认的忽略索引）
        )

        # NaN 守卫：如果损失值变为非有限值（NaN 或 Inf），立即中止训练
        # 这通常表示学习率过高、梯度爆炸或数值不稳定
        if not torch.isfinite(loss):
            print(f"ERROR: Non-finite loss ({loss.item()}) at step {step}. Aborting.")
            # 保存一个紧急检查点以便后续分析问题
            save_checkpoint(model, optimizer, step, float("nan"), checkpoint_dir)
            break                                      # 跳出训练循环

        # 损失除以梯度累积步数 —— 这样多次 backward() 累积的梯度等于单次大批次的效果
        loss = loss / grad_accum
        # 反向传播：计算梯度（梯度会被 .grad 属性累积，不会被清零）
        loss.backward()

        # --- 梯度累积与优化器更新 ---
        # 只有当微步数是梯度累积步数的整数倍时，才执行优化器更新
        if step % grad_accum == 0:
            opt_step += 1                              # 优化器步数递增

            # --- 学习率调度：余弦退火 + 线性预热 ---
            # 预热阶段：学习率从 0 线性增长到 base_lr
            if opt_step < warmup_steps:
                lr = base_lr * opt_step / warmup_steps  # 线性插值：lr = base_lr × (当前步/预热步)
            else:
                # 余弦退火阶段：从 base_lr 余弦衰减到 min_lr
                # progress: 0 → 1，表示在退火阶段中的完成比例
                progress = (opt_step - warmup_steps) / max(1, max_steps - warmup_steps)
                # 标准余弦退火公式：lr = min_lr + (base_lr - min_lr) × 0.5 × (1 + cos(π × progress))
                # cos(π × 0) = 1 → lr = base_lr（退火开始）
                # cos(π × 1) = -1 → lr = min_lr（退火结束）
                lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            # 将计算出的学习率应用到所有参数组
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr                # 更新每个参数组的学习率

            # 梯度裁剪：限制梯度的 L2 范数，防止梯度爆炸
            # grad_clip=0 表示不裁剪（默认行为）
            if train_cfg.get("grad_clip", 0) > 0:
                # clip_grad_norm_ 就地修改梯度，将其范数限制在 grad_clip 以内
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()                           # 执行一步参数更新（AdamW 更新规则）
            optimizer.zero_grad()                      # 清空梯度，为下一轮累积做准备

        # 累积总损失（还原为原始尺度，因为除以了 grad_accum）
        total_loss += loss.item() * grad_accum

        # --- 日志输出 ---
        log_interval = train_cfg.get("log_interval", 50)  # 每隔多少微步打印一次日志，默认 50
        if step % log_interval == 0:
            avg_loss = total_loss / log_interval       # 计算日志区间内的平均损失
            last_avg_loss = avg_loss                   # 记录最近平均损失（用于检查点元数据）
            elapsed = time.time() - t0                 # 本日志区间经过的时间（秒）
            # 计算吞吐量：tokens/秒 = (微步数 × 序列长度 × 批次大小) / 耗时
            # 这里忽略梯度累积的影响，只统计微步，因为每个微步都消耗了这么多 token
            tokens_per_sec = (log_interval * model_cfg.ctx_len * batch_size) / elapsed

            print(
                f"step {step:6d}/{max_steps} | "       # 当前微步 / 总步数，6位右对齐
                f"loss {avg_loss:.4f} | "               # 平均损失，保留4位小数
                f"lr {lr:.2e} | "                       # 当前学习率，科学记数法
                f"{tokens_per_sec:.0f} tok/s | "       # 吞吐量，整数显示
                f"{elapsed:.1f}s"                       # 本区间耗时，保留1位小数
            )

            total_loss = 0.0                           # 重置累积损失，开始新的日志区间
            t0 = time.time()                           # 重置时间起点

            if avg_loss < best_loss:                    # 如果当前损失优于历史最佳
                best_loss = avg_loss                   # 更新最佳损失记录

        # --- 保存检查点 ---
        save_interval = train_cfg.get("save_interval", 1000)  # 每隔多少微步保存一次检查点，默认 1000
        if step % save_interval == 0:
            # 保存编号检查点（如 checkpoint_002000.pt）
            save_checkpoint(model, optimizer, step, last_avg_loss,
                          checkpoint_dir)
            # 同时覆盖保存 latest.pt，方便通过 --resume 快速恢复最新状态
            latest_path = os.path.join(checkpoint_dir, "latest.pt")
            checkpoint = {
                "step": step,                          # 当前步数
                "loss": last_avg_loss,                 # 最近的平均损失
                "model_state_dict": model.state_dict(),  # 模型权重
                "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态
                "config": {k: v for k, v in model.config.__dict__.items()
                          if not k.startswith("_")},   # 配置字典（过滤掉私有属性，即以下划线开头的键）
            }
            torch.save(checkpoint, latest_path)        # 保存到 latest.pt

    # ==================== 训练结束：最终保存 ====================
    # 保存最终模型（只保存模型权重和配置，不保存优化器状态以减小文件体积）
    final_path = os.path.join(checkpoint_dir, "final.pt")
    torch.save({
        "step": step,                                 # 最终步数
        "model_state_dict": model.state_dict(),        # 最终模型权重
        "config": {k: v for k, v in model.config.__dict__.items() if not k.startswith("_")},  # 模型配置
    }, final_path)
    print(f"\nTraining complete! Final model: {final_path}")  # 打印最终模型路径
    print(f"Best loss: {best_loss:.4f}")               # 打印训练过程中的最佳损失


if __name__ == "__main__":
    # 当脚本作为主程序运行时（而非被导入），执行以下代码
    # 创建命令行参数解析器，描述说明这是 RWKV-Courage 语言模型的训练脚本
    parser = argparse.ArgumentParser(description="Train RWKV-Courage language model")
    parser.add_argument("--config", default="configs/courage_25m.yaml",
                        help="Path to YAML config file")  # --config 参数：YAML 配置文件路径，默认为 25M 参数模型配置
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint path")  # --resume 参数：从指定检查点恢复训练
    args = parser.parse_args()                         # 解析命令行参数

    train(config_path=args.config, resume_from=args.resume)  # 启动训练
