#!/usr/bin/env python3  # Shebang：指定使用 python3 解释器执行此脚本
"""  # 模块文档字符串开始
Spike 001: GSA vs RWKV-7 training comparison.  # 实验代号：GSA 与 RWKV-7 的训练对比
  # 空行（文档字符串内格式分隔）
Trains both models on identical synthetic data for 100 steps.  # 在相同的合成数据上分别训练两个模型，各跑 100 步
Compares: loss curve, training speed, parameter count.  # 对比指标：损失曲线、训练速度、参数量
  # 空行（文档字符串内格式分隔）
Usage:  # 用法说明标题
    python compare.py [--steps 100] [--ctx 128] [--batch 2]  # 命令行调用示例及可选参数
"""  # 模块文档字符串结束

  # 空行（代码分节）
import argparse  # 导入 argparse 模块，用于解析命令行参数
import os  # 导入 os 模块，用于文件路径拼接等操作系统接口
import sys  # 导入 sys 模块，用于操作 sys.modules 实现动态模块注册
import time  # 导入 time 模块，用于计时训练耗时
import types  # 导入 types 模块，用于动态创建模块对象（ModuleType）
import importlib.util  # 导入 importlib.util，用于从文件路径动态加载模块
from pathlib import Path  # 从 pathlib 导入 Path，用于面向对象的路径操作

  # 空行（代码分节）
# Path setup  # 路径配置注释
SPIKE_DIR = Path(__file__).parent  # 获取当前脚本所在目录（models/gsa/）的 Path 对象
PROJECT_DIR = SPIKE_DIR.parent.parent  # 向上两级获取项目根目录（models/ → subq-models/）
SRC_DIR = str(PROJECT_DIR / "rwkv-courage" / "src")  # 拼接出 RWKV-7 源码目录的绝对路径字符串

  # 空行（代码分节）
import torch  # 导入 PyTorch 深度学习框架
import torch.nn.functional as F  # 导入 PyTorch 的函数式神经网络接口（如 cross_entropy）

  # 空行（代码分节）
# Import RWKV-7 model (handles relative import from src/ package)  # 动态导入 RWKV-7 模型（处理 src/ 包内的相对导入）
spec_wkv = importlib.util.spec_from_file_location(  # 根据文件路径创建模块加载规格（ModuleSpec）
    "wkv7_operator", os.path.join(SRC_DIR, "wkv7_operator.py"))  # 模块名为 "wkv7_operator"，源文件为 src/wkv7_operator.py
wkv7 = importlib.util.module_from_spec(spec_wkv)  # 根据规格创建空的模块对象（尚未执行代码）
sys.modules["wkv7_operator"] = wkv7  # 将模块对象注册到 sys.modules，使后续 import 能找到它
spec_wkv.loader.exec_module(wkv7)  # 执行模块代码，填充 wkv7 模块（此时模块内部的相对 import 可能仍需处理）

  # 空行（代码分节）
# Create fake package for src/ to satisfy relative imports  # 创建假的 src 包，以满足 RWKV-7 源码中的相对导入
src_pkg = types.ModuleType("src")  # 创建一个名为 "src" 的空白模块对象（模拟 Python 包）
src_pkg.wkv7_operator = wkv7  # 将已加载的 wkv7_operator 模块挂载到 src 包的命名空间中
sys.modules["src"] = src_pkg  # 注册 src 包到 sys.modules，使 "from src.wkv7_operator import ..." 能工作
sys.modules["src.wkv7_operator"] = wkv7  # 同时注册完整路径，兼容不同的导入写法

  # 空行（代码分节）
spec_model = importlib.util.spec_from_file_location(  # 为 src/model.py 创建模块加载规格
    "src.model", os.path.join(SRC_DIR, "model.py"))  # 模块名为 "src.model"，源文件为 src/model.py
model_mod = importlib.util.module_from_spec(spec_model)  # 根据规格创建空的模型模块对象
sys.modules["src.model"] = model_mod  # 将模型模块注册到 sys.modules，供内部导入解析
spec_model.loader.exec_module(model_mod)  # 执行 model.py 的代码，填充 CourageLM 和 RWKV7Config 类

  # 空行（代码分节）
CourageLM = model_mod.CourageLM  # 从动态加载的模块中取出 CourageLM 类（RWKV-7 的 LM 模型）
RWKV7Config = model_mod.RWKV7Config  # 从动态加载的模块中取出 RWKV7Config 类（RWKV-7 配置）

  # 空行（代码分节）
from model_gsa import GSA_Config, GSALanguageModel  # 从本地 model_gsa.py 导入 GSA 的配置类和语言模型类

  # 空行（代码分节＋空行）

def build_configs(vocab_size=8000, ctx_len=128):  # 构建 RWKV-7 和 GSA 的匹配配置（默认词表 8000，上下文 128）
    """Build matching configs for fair comparison."""  # 文档字符串：为公平对比创建匹配的配置
    base = {  # 创建基础参数字典，两种模型共享的核心超参数
        "n_embd": 384,  # 嵌入维度：384（小模型用于快速 spike 实验）
        "n_layer": 8,  # Transformer/RNN 层数：8 层
        "vocab_size": vocab_size,  # 词表大小，从函数参数传入，默认 8000
        "ctx_len": ctx_len,  # 上下文长度，从函数参数传入，默认 128
        "n_head": 6,  # 注意力头数：6 个头
        "head_dim": 64,  # 每个注意力头的维度：64（n_head × head_dim = 384 = n_embd）
        "global_k": 64,  # GSA 全局注意力中 kv 压缩后的 token 数：64
        "window": 32,  # GSA 局部滑动窗口大小：32 个 token
    }  # 基础参数字典结束

  # 空行（代码分节）
    rwkv_cfg = RWKV7Config({  # 使用基础参数创建 RWKV-7 配置对象（传入字典）
        "n_embd": base["n_embd"],  # 嵌入维度：与 GSA 保持一致 (384)
        "n_layer": base["n_layer"],  # 层数：与 GSA 保持一致 (8)
        "vocab_size": base["vocab_size"],  # 词表大小：与 GSA 保持一致
        "ctx_len": base["ctx_len"],  # 上下文长度：与 GSA 保持一致
        "head_size_a": base["head_dim"],  # RWKV-7 的头大小 a：映射到 head_dim (64)
        "D_DECAY_LORA": 32,  # RWKV-7 特有的衰减 LoRA 维度：32
        "D_AAA_LORA": 32,  # RWKV-7 特有的 AAA LoRA 维度：32
        "D_MV_LORA": 16,  # RWKV-7 特有的 MV LoRA 维度：16
        "D_GATE_LORA": 64,  # RWKV-7 特有的门控 LoRA 维度：64
    })  # RWKV-7 配置对象创建结束

  # 空行（代码分节）
    gsa_cfg = GSA_Config(base)  # 使用基础字典创建 GSA 配置对象（GSA_Config 直接接受 dict）
    return rwkv_cfg, gsa_cfg  # 返回两个配置对象：RWKV-7 配置、GSA 配置

  # 空行（代码分节＋空行）

def train_steps(model, optimizer, steps, ctx_len, batch_size, device, dtype, label=""):  # 训练函数：对指定模型执行 steps 步训练
    """Run training steps, return list of (step, loss, elapsed)."""  # 文档字符串：执行训练步骤，返回 (步数, 损失, 耗时) 列表
    model.train()  # 将模型设置为训练模式（启用 dropout、batch norm 等）
    records = []  # 初始化记录列表，用于存储每步的 (步数, 损失值, 累计秒数)
    t0 = time.time()  # 记录训练开始时间戳（秒），用于计算累计耗时

  # 空行（代码分节）
    # Pre-generate synthetic data (same for both models)  # 预生成合成训练数据（两个模型使用相同数据以保证公平）
    torch.manual_seed(42)  # 设置 PyTorch 全局随机种子为 42，确保每次运行生成相同数据（可复现）
    # Handle different config access patterns  # 处理不同模型的配置访问方式差异
    if hasattr(model, 'cfg'):  # 如果模型通过 .cfg 属性访问配置（如 GSA 模型）
        vocab_size = model.cfg.vocab_size  # 从 model.cfg 读取词表大小
    else:  # 否则（如 RWKV-7 模型可能用 .config 属性）
        vocab_size = model.config.vocab_size  # 从 model.config 读取词表大小
    data = torch.randint(0, vocab_size, (steps * batch_size, ctx_len + 1))  # 生成随机整数张量作为合成数据：形状 (总样本数, 序列长度+1)，值域 [0, vocab_size)

  # 空行（代码分节）
    for step in range(steps):  # 遍历每一步训练（0 到 steps-1）
        # Batch  # 批量构造注释
        x_batch = torch.zeros(batch_size, ctx_len, dtype=torch.long, device=device)  # 初始化输入张量：形状 (batch, ctx_len)，long 类型，放在指定设备上
        y_batch = torch.zeros(batch_size, ctx_len, dtype=torch.long, device=device)  # 初始化目标张量：形状 (batch, ctx_len)，long 类型，放在指定设备上
        for b in range(batch_size):  # 遍历批次中的每个样本
            idx = step * batch_size + b  # 计算该样本在预生成数据中的行索引
            x_batch[b] = data[idx, :ctx_len]  # 取 data 行的前 ctx_len 列作为输入（token 0 到 ctx_len-1）
            y_batch[b] = data[idx, 1:ctx_len+1]  # 取 data 行的后 ctx_len 列作为目标（token 1 到 ctx_len，即下一个 token 预测）

  # 空行（代码分节）
        optimizer.zero_grad()  # 清空优化器中的梯度缓存（否则梯度会累积）
        logits = model(x_batch)  # 前向传播：输入 x_batch，输出 logits（未归一化的 log 概率）
        loss = F.cross_entropy(  # 计算交叉熵损失函数
            logits.view(-1, logits.shape[-1]),  # 将 logits 展平为 (batch*ctx_len, vocab_size) 形状
            y_batch.view(-1),  # 将目标标签展平为 (batch*ctx_len,) 一维张量（类别索引）
        )  # 交叉熵计算结束，返回标量损失值
        loss.backward()  # 反向传播：计算损失对模型所有参数的梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪：将梯度的全局 L2 范数限制在 1.0 以内，防止梯度爆炸
        optimizer.step()  # 优化器更新：使用 AdamW 算法更新模型参数

  # 空行（代码分节）
        records.append((step + 1, loss.item(), time.time() - t0))  # 记录当前步骤：(步数从1开始, 损失标量值, 从开始到现在的累计秒数)

  # 空行（代码分节）
    return records  # 返回所有训练步骤的记录列表

  # 空行（代码分节＋空行）

def main():  # 主函数：解析参数、构建模型、训练、输出对比结果
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument("--steps", type=int, default=100)  # 添加 --steps 参数：训练步数，整数类型，默认 100
    parser.add_argument("--ctx", type=int, default=128)  # 添加 --ctx 参数：上下文长度，整数类型，默认 128
    parser.add_argument("--batch", type=int, default=2)  # 添加 --batch 参数：批次大小，整数类型，默认 2
    parser.add_argument("--device", default="cpu")  # 添加 --device 参数：计算设备，默认为 cpu（安全起见 spike 阶段用 CPU）  # use cpu for spike safety
    args = parser.parse_args()  # 解析命令行参数，返回命名空间对象

  # 空行（代码分节）
    # Detect device  # 设备检测逻辑注释
    if args.device == "auto":  # 如果用户选择自动检测设备
        if torch.backends.mps.is_available():  # 检查 Apple Metal Performance Shaders 是否可用（macOS）
            device = torch.device("mps")  # 可用则选择 MPS 设备（Apple GPU 加速）
        elif torch.cuda.is_available():  # 检查 NVIDIA CUDA 是否可用
            device = torch.device("cuda")  # 可用则选择 CUDA 设备（NVIDIA GPU）
        else:  # 两者都不可用
            device = torch.device("cpu")  # 回退到 CPU
    else:  # 用户显式指定了设备名
        device = torch.device(args.device)  # 直接使用用户指定的设备

  # 空行（代码分节）
    dtype = torch.float32  # 计算精度设为 float32（单精度浮点）
    vocab_size = 8000  # 合成数据的词表大小固定为 8000
    ctx_len = args.ctx  # 从命令行参数获取上下文长度
    batch_size = args.batch  # 从命令行参数获取批次大小
    steps = args.steps  # 从命令行参数获取训练步数

  # 空行（代码分节）
    print(f"=== Spike 001: GSA vs RWKV-7 ===")  # 打印实验标题
    print(f"Device: {device} | ctx={ctx_len} | batch={batch_size} | steps={steps}")  # 打印实验配置信息
    print(f"MPS available: {torch.backends.mps.is_available()}")  # 打印 MPS 是否可用（调试用）
    print()  # 打印空行，美化输出

  # 空行（代码分节）
    # Build models  # 构建模型注释
    rwkv_cfg, gsa_cfg = build_configs(vocab_size, ctx_len)  # 调用 build_configs 获取两个模型的匹配配置

  # 空行（代码分节）
    rwkv_model = CourageLM(rwkv_cfg).to(device).to(dtype)  # 实例化 RWKV-7 模型，移动到指定设备并转换为指定精度
    gsa_model = GSALanguageModel(gsa_cfg).to(device).to(dtype)  # 实例化 GSA 模型，移动到指定设备并转换为指定精度

  # 空行（代码分节）
    print(f"RWKV-7 params: {rwkv_model.count_parameters():,} ({rwkv_model.count_parameters()/1e6:.1f}M)")  # 打印 RWKV-7 参数量（带千分位分隔符及百万单位）
    print(f"GSA params:   {gsa_model.count_parameters():,} ({gsa_model.count_parameters()/1e6:.1f}M)")  # 打印 GSA 参数量（带千分位分隔符及百万单位）
    print()  # 打印空行

  # 空行（代码分节）
    # Identical optimizers  # 使用相同的优化器配置以确保公平对比
    rwkv_opt = torch.optim.AdamW(rwkv_model.parameters(), lr=3e-4)  # 为 RWKV-7 创建 AdamW 优化器，学习率 3e-4
    gsa_opt = torch.optim.AdamW(gsa_model.parameters(), lr=3e-4)  # 为 GSA 创建 AdamW 优化器，学习率 3e-4

  # 空行（代码分节）
    # --- Train RWKV-7 ---  # 训练 RWKV-7 的分节标题
    print("Training RWKV-7...")  # 打印训练开始提示
    rwkv_records = train_steps(rwkv_model, rwkv_opt, steps, ctx_len, batch_size,  # 调用 train_steps 训练 RWKV-7（参数：模型、优化器、步数、上下文长度、批次大小）
                               device, dtype, "RWKV-7")  # 传入设备、精度、标签 "RWKV-7"（参数续行）

  # 空行（代码分节）
    # --- Train GSA ---  # 训练 GSA 的分节标题
    print("Training GSA...")  # 打印训练开始提示
    gsa_records = train_steps(gsa_model, gsa_opt, steps, ctx_len, batch_size,  # 调用 train_steps 训练 GSA（参数：模型、优化器、步数、上下文长度、批次大小）
                              device, dtype, "GSA")  # 传入设备、精度、标签 "GSA"（参数续行）

  # 空行（代码分节）
    # --- Report ---  # 结果报告分节标题
    print()  # 打印空行
    print("=" * 70)  # 打印 70 个等号作为标题分隔线
    print("RESULTS")  # 打印 "RESULTS" 标题
    print("=" * 70)  # 打印 70 个等号作为标题下划线

  # 空行（代码分节）
    rwkv_start = rwkv_records[0][1]  # 取 RWKV-7 第 1 步的损失值（索引 0 的元素 [1]）
    rwkv_end = rwkv_records[-1][1]  # 取 RWKV-7 最后一步的损失值（索引 -1 的元素 [1]）
    rwkv_time = rwkv_records[-1][2]  # 取 RWKV-7 的总训练耗时（最后一条记录的累计秒数）

  # 空行（代码分节）
    gsa_start = gsa_records[0][1]  # 取 GSA 第 1 步的损失值
    gsa_end = gsa_records[-1][1]  # 取 GSA 最后一步的损失值
    gsa_time = gsa_records[-1][2]  # 取 GSA 的总训练耗时

  # 空行（代码分节）
    print(f"{'':20} {'RWKV-7':>12} {'GSA':>12} {'Ratio':>10}")  # 打印对比表格表头：指标名(左对齐20), RWKV-7(右对齐12), GSA(右对齐12), 比值(右对齐10)
    print(f"{'─'*20} {'─'*12} {'─'*12} {'─'*10}")  # 打印表格分隔线（Unicode 框线字符 ─）
    print(f"{'Params':20} {rwkv_model.count_parameters():>12,} {gsa_model.count_parameters():>12,} {gsa_model.count_parameters()/rwkv_model.count_parameters():>9.2f}x")  # 参数量行：RWKV 参数量、GSA 参数量、GSA/RWKV 比值（倍数）
    print(f"{'Step 1 loss':20} {rwkv_start:>12.4f} {gsa_start:>12.4f}")  # 第一步损失行：RWKV 和 GSA 的初始损失值（4 位小数）
    print(f"{'Step N loss':20} {rwkv_end:>12.4f} {gsa_end:>12.4f}")  # 最后一步损失行：RWKV 和 GSA 的最终损失值
    print(f"{'Loss reduction':20} {rwkv_start - rwkv_end:>11.4f} {gsa_start - gsa_end:>11.4f}")  # 损失下降量行：初始损失减去最终损失（值越大收敛越多）
    print(f"{'Total time':20} {rwkv_time:>11.1f}s {gsa_time:>11.1f}s {gsa_time/rwkv_time:>9.2f}x")  # 总耗时行：RWKV 耗时、GSA 耗时、时间比值（倍数）
    print(f"{'Time/step':20} {rwkv_time/steps*1000:>9.0f}ms {gsa_time/steps*1000:>9.0f}ms")  # 每步耗时行：RWKV 平均每步毫秒数、GSA 平均每步毫秒数

  # 空行（代码分节）
    # Per-step loss trace (first 10 + last)  # 逐步损失追踪（显示前 10 步 + 最后若干步）
    print()  # 打印空行
    print(f"{'Step':>6} {'RWKV-7 loss':>12} {'GSA loss':>12}")  # 打印损失追踪表头：步数、RWKV-7 损失、GSA 损失
    for i in range(min(10, len(rwkv_records))):  # 遍历前 min(10, 总步数) 条记录（最多显示前 10 步）
        s_r, l_r, _ = rwkv_records[i]  # 解包 RWKV-7 的第 i 条记录：(步数, 损失, _)，下划线忽略耗时
        s_g, l_g, _ = gsa_records[i]  # 解包 GSA 的第 i 条记录：(步数, 损失, _)
        print(f"{s_r:>6} {l_r:>12.4f} {l_g:>12.4f}")  # 打印一行对比：步数、RWKV 损失、GSA 损失（对齐格式）

  # 空行（代码分节）
    if steps > 10:  # 如果总步数超过 10 步（前面已显示头 10 步）
        print(f"  ...")  # 打印省略号表示中间步骤被省略
    if steps > 20:  # 如果总步数超过 20 步（既省略了中间，又显示最后）
        # Last 5 steps  # 显示最后 5 步的注释
        for i in range(max(0, steps - 5), steps):  # 遍历最后 5 步的索引范围 [steps-5, steps)
            s_r, l_r, _ = rwkv_records[i]  # 解包 RWKV-7 记录
            s_g, l_g, _ = gsa_records[i]  # 解包 GSA 记录
            print(f"{s_r:>6} {l_r:>12.4f} {l_g:>12.4f}")  # 打印最后几步的损失对比

  # 空行（代码分节）
    print()  # 打印空行
    print("Done. See README.md for verdict.")  # 打印完成提示：查看 README.md 获得结论

  # 空行（代码分节＋空行）

if __name__ == "__main__":  # Python 标准入口守卫：仅当脚本直接运行时（非 import）才执行 main
    main()  # 调用主函数，启动整个对比实验流程
