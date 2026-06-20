#!/usr/bin/env python3
# Shebang：指定用 python3 解释器执行此脚本，使得在 Unix 系统上可以直接 ./demo_chat.py 运行
"""
GSA Chat Demo — 加载训练好的 GSA (Gated Slot Attention) 模型进行对话生成。

用法:
    # 模式 1: 加载 checkpoint（从训练保存的检查点文件恢复模型权重）进行推理
    python demo_chat.py --checkpoint checkpoints/final.pt --prompt "I believe"

    # 模式 2: 使用未训练模型测试数据管道（模型权重为随机初始化，输出也将是随机的）
    python demo_chat.py --untrained --prompt "Hello"

    # 模式 3: 交互模式（启动后用户可以反复输入 prompt 与模型对话）
    python demo_chat.py --checkpoint checkpoints/final.pt --interactive

依赖:
    pip install tokenizers  (用于 BPE (Byte-Pair Encoding) tokenizer 分词器，如未安装则自动回退到简单空格分词)
"""

import argparse  # 导入 argparse 模块，用于解析命令行参数（如 --checkpoint、--prompt 等）
import os  # 导入 os 模块，用于检查文件路径是否存在 (os.path.exists) 等操作系统相关操作
import sys  # 导入 sys 模块，用于程序异常退出 (sys.exit) 等系统级操作
import types  # 导入 types 模块，用于类型相关操作（此脚本中虽已导入但未显式使用，可能是预留或历史遗留）
import importlib.util  # 导入 importlib.util 模块，用于动态导入功能（如 spec_from_file_location，此脚本中亦未显式使用，可能是预留）
from pathlib import Path  # 从 pathlib 模块导入 Path 类，用于跨平台的路径操作，比 os.path 更现代且可读性更好

# 路径配置：定义项目目录结构，统一管理路径
SPIKE_DIR = Path(__file__).parent  # 获取当前脚本所在目录的绝对路径，即 models/gsa/（spike 实验目录）
PROJECT_DIR = SPIKE_DIR.parent.parent  # 从 SPIKE_DIR 向上两级得到项目根目录（即 models/ 的上一级）
SRC_DIR = str(PROJECT_DIR / "rwkv-courage" / "src")  # 构建 RWKV-Courage 源码目录的路径字符串，用于可能的模块导入

import torch  # 导入 PyTorch 深度学习框架，用于张量运算、模型加载和 GPU 加速

# 从本地模块导入 GSA (Gated Slot Attention) 模型的核心组件
from model_gsa import GSA_Config, GSALanguageModel  # 从同目录下的 model_gsa.py 导入配置类和语言模型类


def get_tokenizer(tokenizer_path: str = None):
    # 函数定义：获取或创建 tokenizer（分词器）
    # 参数 tokenizer_path: 可选的 BPE tokenizer 文件路径（如 tokenizer.json），若为 None 则直接使用简单分词器
    # 返回: (tokenizer 对象, 类型字符串 "BPE" 或 "simple")
    """加载或创建简单的 tokenizer。"""
    if tokenizer_path and os.path.exists(tokenizer_path):
        # 条件判断：如果提供了 tokenizer 路径且该文件存在
        try:
            # 尝试加载 BPE tokenizer
            from tokenizers import Tokenizer
            # 从 HuggingFace tokenizers 库导入 Tokenizer 类（用于 BPE/WordPiece 等子词分词）
            tok = Tokenizer.from_file(tokenizer_path)
            # 从 JSON 文件反序列化加载训练好的 BPE tokenizer
            return tok, "BPE"
            # 返回 BPE tokenizer 对象和类型标识字符串 "BPE"
        except ImportError:
            # 捕获 ImportError：tokenizers 库未安装
            print("[WARN] tokenizers 库未安装，使用简单空格分词器")
            # 打印警告信息，提示用户安装 tokenizers 库以获得更好的分词效果
        except Exception as e:
            # 捕获其他所有异常（如 JSON 格式损坏、版本不兼容等）
            print(f"[WARN] 无法加载 tokenizer: {e}，使用简单空格分词器")
            # 打印具体错误信息，然后回退到简单分词器

    # 如果以上方法都失败或未提供路径，则使用内置的简易分词器作为回退方案
    return SimpleTokenizer(), "simple"
    # 创建 SimpleTokenizer 实例并返回，类型标识为 "simple"


class SimpleTokenizer:
    # 类定义：简易字符级分词器，仅用于 demo 演示目的（不做真正的子词分割）
    """简单字符级分词器，仅用于 demo。
    输出 token ID 在 [0, vocab_size) 范围内，保证所有生成的 token ID 都在有效范围内，防止索引越界。
    """

    def __init__(self, vocab_size: int = 8000):
        # 构造函数，初始化分词器
        # 参数 vocab_size: 词表大小，默认 8000（与 GSA 模型的默认配置保持一致）
        self.vocab_size = vocab_size
        # 保存词表大小到实例属性
        self.pad_id = 0
        # 定义填充 (padding) token 的 ID 为 0（用于将不等长序列补齐到相同长度）
        self.unk_id = 1
        # 定义未知 (unknown) token 的 ID 为 1（用于替代词表中不存在的字符/词）
        self.bos_id = 2
        # 定义序列开始 (Begin Of Sequence) token 的 ID 为 2
        self.eos_id = 3
        # 定义序列结束 (End Of Sequence) token 的 ID 为 3

    def encode(self, text: str) -> list:
        # 编码方法：将文本字符串转换为 token ID 列表
        # 参数 text: 要编码的输入文本字符串
        # 返回: token ID 的 Python 列表，列表首元素为 BOS token
        ids = [self.bos_id]
        # 在序列开头插入 BOS (Begin Of Sequence) token，标记序列开始
        for ch in text:
            # 遍历输入文本中的每一个字符
            # 将字符映射到 [4, vocab_size) 范围内，即词表后四个以上位置
            # 保留前 4 个 ID 给特殊 token (PAD=0, UNK=1, BOS=2, EOS=3)
            tid = 4 + (ord(ch) % (self.vocab_size - 4))
            # ord(ch) 获取 Unicode 码点，对可用范围取模后加 4 偏移
            # 示例：vocab_size=8000 时，范围是 [4, 7999]，vocab_size-4=7996，ord('A')=65 -> tid=4+65=69
            ids.append(min(tid, self.vocab_size - 1))
            # 使用 min 做安全裁剪，确保 token ID 不超过 vocab_size-1（防止取模+偏移后可能的越界）
        return ids
        # 返回完整的 token ID 序列（包括开头的 BOS token）

    def decode(self, ids: list) -> str:
        # 解码方法：将 token ID 列表反向转换为文本字符串
        # 参数 ids: token ID 列表
        # 返回: 解码后的文本字符串（近似还原，因为字符级映射不可逆）
        chars = []
        # 初始化空列表，用于收集解码后的字符
        for i in ids:
            # 遍历输入的每个 token ID
            if i <= 3 or i >= self.vocab_size:
                # 跳过特殊 token（PAD=0, UNK=1, BOS=2, EOS=3）和超出词表范围的无效 ID
                continue
                # 不输出这些 ID 对应的内容，继续下一个
            # 逆映射回可打印字符（因为 ord->mod 不可逆，所以只能近似还原）
            c = chr(32 + (i - 4) % 95)
            # 减去偏移 4 后对 95 取模（ASCII 可打印字符 32-126 共 95 个），加 32 从空格开始
            # 设计决策：由于 encode 使用 mod 映射，不同字符可能映射到同一个 ID，因此 decode 不可逆
            chars.append(c)
            # 将解码出的字符加入列表
        return ''.join(chars)
        # 将所有字符合并为一个字符串返回


def load_model(checkpoint_path: str, device: torch.device):
    # 函数定义：从 checkpoint 文件加载训练好的 GSA 模型
    # 参数 checkpoint_path: checkpoint 文件的路径（.pt 文件，PyTorch 序列化格式）
    # 参数 device: torch.device 对象，指定模型加载到哪个设备（cpu、cuda:0、mps 等）
    # 返回: (model 模型对象, step 训练步数字符串, loss 损失值字符串)，后两者用于显示
    """从 checkpoint 加载模型。"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # 使用 PyTorch 加载 checkpoint 文件
    # map_location=device: 将张量直接加载到目标设备，避免先加载到 CPU 再复制到设备的内存浪费
    # weights_only=False: 允许加载任意 Python 对象（如 config 字典），不仅仅是权重张量
    # PyTorch 2.6+ 默认 weights_only=True，此处显式关闭以保证兼容性

    # 从 checkpoint 中恢复模型配置（超参数）
    if "config" in ckpt:
        # 如果 checkpoint 中保存了 config 字典（推荐做法）
        cfg_dict = ckpt["config"]
        # 直接读取保存的配置字典
    else:
        # 如果 checkpoint 中没有保存 config（兼容旧版本或外部模型）
        # 使用默认的 spike 实验模型超参数配置
        cfg_dict = {
            "n_embd": 384, "n_layer": 8, "vocab_size": 8000,
            # 嵌入维度 384，8 层 Transformer，词表大小 8000
            "ctx_len": 4096, "n_head": 6, "head_dim": 64,
            # 上下文长度 4096（最大序列长度），6 个注意力头，每个头维度 64 -> 总维度 6*64=384
            "global_k": 64, "window": 32,
            # GSA (Gated Slot Attention) 特有参数：全局注意力保留前 64 个 slot，局部窗口大小 32
        }

    cfg = GSA_Config(cfg_dict)
    # 用配置字典实例化 GSA 配置对象（包含参数验证和默认值填充）
    model = GSALanguageModel(cfg).to(device)
    # 用配置创建 GSALanguageModel 模型实例，并立即移动到目标设备（CPU/GPU/MPS）

    # 加载模型权重参数
    if "model_state_dict" in ckpt:
        # 如果 checkpoint 中存的是包含元信息的字典（推荐格式：包含 model_state_dict/step/loss）
        model.load_state_dict(ckpt["model_state_dict"])
        # 从 state_dict 加载模型权重到刚创建的模型实例中
        step = ckpt.get("step", "?")
        # 读取训练步数，若不存在则显示 "?"（标记为未知）
        loss = ckpt.get("loss", "?")
        # 读取训练损失值，若不存在则显示 "?"（标记为未知）
    else:
        # 如果 checkpoint 直接就是 state_dict（简化格式，不含元信息）
        model.load_state_dict(ckpt)
        # 直接将整个 checkpoint 视为 state_dict 加载权重
        step, loss = "?", "?"
        # 无法获取步数和损失，设为未知

    model.eval()
    # 将模型切换到评估模式 (eval mode)
    # eval() 的作用：关闭 Dropout、BatchNorm 等仅在训练时生效的层，确保推理时行为确定且充分利用已学参数
    return model, step, loss
    # 返回模型对象、训练步数、损失值


def create_untrained_model(device: torch.device):
    # 函数定义：创建一个未训练的（随机权重初始化）GSA 模型
    # 参数 device: torch.device 对象，指定模型所在的设备
    # 返回: 未训练的 GSALanguageModel 模型实例（权重为 PyTorch 默认的随机初始化值）
    """创建未训练的 GSA 模型（用于管道测试）。"""
    cfg = GSA_Config({
        # 使用与 load_model 中 fallback 相同的默认超参数创建配置对象
        "n_embd": 384, "n_layer": 8, "vocab_size": 8000,
        # 嵌入维度 384（较小的模型便于快速测试），8 层 Transformer，词表 8000
        "ctx_len": 4096, "n_head": 6, "head_dim": 64,
        # 上下文窗口 4096 token，6 头注意力，每头 64 维 = 总嵌入维度 384
        "global_k": 64, "window": 32,
        # GSA 参数：64 个全局 slot，32 窗口大小的局部注意力
    })
    model = GSALanguageModel(cfg).to(device)
    # 创建模型并移动到目标设备，此时权重由 PyTorch 的 nn.Module 默认初始化（如 Kaiming 初始化等）
    model.eval()
    # 切换到评估模式（虽然未训练，但保持一致的接口行为）
    return model
    # 返回未训练的模型


def generate_response(model, tokenizer, prompt: str, device,
                      max_tokens: int = 100, temperature: float = 0.8):
    # 函数定义：根据 prompt 使用模型生成回复文本
    # 参数 model: GSA 语言模型实例（已训练或未训练）
    # 参数 tokenizer: 分词器对象（BPE Tokenizer 或 SimpleTokenizer），需支持 encode 和 decode 方法
    # 参数 prompt: 用户输入的起始文本，模型将从此处开始续写
    # 参数 device: torch.device 对象，指定张量计算在哪个设备上进行
    # 参数 max_tokens: 最大生成 token 数，默认 100，控制回复的长度上限
    # 参数 temperature: 采样温度，默认 0.8。温度越高生成越随机（多样化），越低越确定（保守）。范围通常 (0, 2]
    # 返回: 模型生成的续写文本字符串（不包含原始 prompt）
    """生成对 prompt 的续写。"""
    if hasattr(tokenizer, 'encode'):
        # 检查 tokenizer 是否具备 encode 方法（标准分词器接口）
        if isinstance(tokenizer, SimpleTokenizer):
            # 如果是 SimpleTokenizer（简单字符级分词器）
            ids = tokenizer.encode(prompt)
            # 直接调用 encode 方法，内部已包含 BOS token 添加逻辑
        else:
            # 否则是 BPE tokenizer 或其他标准分词器
            encoded = tokenizer.encode(prompt)
            # 调用 encode 方法，返回 Encoding 对象（包含 .ids 属性）
            ids = encoded.ids
            # 从 Encoding 对象提取 token ID 列表
            # BPE 分词器可能不自动添加 BOS，这里手动补上
            try:
                # 尝试获取并添加 BOS token
                bos_id = tokenizer.token_to_id("[BOS]")
                # 在词表中查找 "[BOS]" 对应的 token ID
                if bos_id:
                    # 如果找到 BOS token ID（不为 None 且不为 0）
                    ids = [bos_id] + ids
                    # 在 token ID 列表开头插入 BOS token
            except Exception:
                # 如果 token_to_id 方法不可用或抛出异常（如某些简易分词器）
                pass
                # 静默忽略，不中断程序
                # 设计决策：BOS token 添加失败不影响生成结果，仅在部分模型中略微影响开头生成质量
    else:
        # 如果 tokenizer 完全没有 encode 方法（极端回退情况）
        ids = [ord(c) for c in prompt]
        # 直接将每个字符的 Unicode 码点作为 token ID（非常简陋的回退方案）

    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    # 将 token ID 列表转换为 PyTorch 张量
    # [ids] 外面套一层列表：将 1D 序列包装成 batch_size=1 的 2D 张量，形状为 (1, seq_len)
    # dtype=torch.long: token ID 必须是长整型（int64），因为嵌入层索引需要整数
    # device=device: 张量直接创建在目标设备上，避免后续的 .to(device) 调用
    output_tensor = model.generate(
        # 调用 GSA 模型的 generate 方法执行自回归生成
        input_tensor,
        # 输入的上下文张量（prompt 的 token 序列）
        max_new_tokens=max_tokens,
        # 最多生成的新 token 数量（不包含输入的 prompt 长度）
        temperature=temperature,
        # 采样温度，控制 softmax 概率分布的锐度
        # 工作原理：temperature -> 0 时趋于 argmax（确定性），temperature -> 无穷时趋于均匀分布（完全随机）
    )

    output_ids = output_tensor[0].tolist()[len(ids):]
    # 提取纯生成的 token ID（去掉原始 prompt 部分）
    # output_tensor[0]: 取 batch 中第一个（也是唯一一个）样本，形状从 (1, total_len) 变为 (total_len,)
    # .tolist(): 将 PyTorch 张量转换为 Python 列表
    # [len(ids):]: 切片操作，跳过输入 prompt 的长度，只保留模型新生成的部分

    if hasattr(tokenizer, 'decode'):
        # 检查 tokenizer 是否具备 decode 方法
        text = tokenizer.decode(output_ids)
        # 调用分词器的 decode 方法将 token ID 列表还原为文本
    else:
        # 如果 tokenizer 没有 decode 方法（极端回退情况）
        text = ''.join(chr(i) if 32 <= i < 127 else ' ' for i in output_ids)
        # 将 token ID 映射到 ASCII 可打印字符：32 <= i < 127 取 ASCII 可打印范围（空格到波浪号），超出替换为空格
        # 设计决策：这是最后的回退方案，仅保证程序不崩溃，输出质量无法保证
        # 正常情况下不会走到这个分支，因为 SimpleTokenizer 和 BPE tokenizer 都实现了 decode

    return text
    # 返回解码后的生成文本


def main():
    # 主函数：程序入口，负责解析参数、加载模型、选择模式并执行对话
    parser = argparse.ArgumentParser(description="GSA Chat Demo")
    # 创建命令行参数解析器，description 在 --help 时显示
    parser.add_argument("--checkpoint", help="模型 checkpoint 路径")
    # 添加 --checkpoint 参数：指定训练好的模型检查点文件路径
    parser.add_argument("--untrained", action="store_true",
                        help="使用未训练模型（管道测试）")
    # 添加 --untrained 参数：布尔标志，存在即为 True
    # action="store_true": 当命令行中出现此标志时，args.untrained 为 True，否则为 False
    parser.add_argument("--tokenizer", help="BPE tokenizer.json 路径")
    # 添加 --tokenizer 参数：指定 BPE 分词器配置文件的路径
    parser.add_argument("--prompt", default="I believe",
                        help="起始 prompt")
    # 添加 --prompt 参数：默认起始文本为 "I believe"，模型将从此文本开始续写
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="最大生成 token 数")
    # 添加 --max-tokens 参数：整型，默认生成 100 个 token，控制生成文本的长度上限
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="采样温度")
    # 添加 --temperature 参数：浮点型，默认 0.8，控制生成的随机性/创造性
    parser.add_argument("--interactive", action="store_true",
                        help="交互对话模式")
    # 添加 --interactive 参数：布尔标志，启用交互对话模式，程序保持运行，用户可反复输入 prompt
    parser.add_argument("--device", default="cpu",
                        help="设备: cpu, mps, cuda")
    # 添加 --device 参数：默认使用 CPU，支持 CPU、Apple Silicon Metal (MPS)、NVIDIA GPU (CUDA)
    args = parser.parse_args()
    # 解析命令行参数，返回包含所有参数的 Namespace 对象

    # 设备选择逻辑：根据用户参数和硬件可用性选择计算设备
    if args.device == "auto" or args.device == "mps":
        # 如果用户指定 "auto" 自动选择，或指定了 "mps"（Apple Silicon）
        if torch.backends.mps.is_available():
            # 优先检查 MPS (Apple Metal Performance Shaders) 是否可用（macOS 上的 GPU 加速）
            device = torch.device("mps")
            # 创建 MPS 设备对象
        elif torch.cuda.is_available():
            # 如果 MPS 不可用但 CUDA 可用（NVIDIA GPU）
            device = torch.device("cuda")
            # 创建 CUDA 设备对象（默认 cuda:0，即第一块 GPU）
        else:
            # MPS 和 CUDA 都不可用
            device = torch.device("cpu")
            # 回退到 CPU（最通用的选项，总是可用）
    else:
        # 用户指定了具体设备（如 "cpu"、"cuda:1" 等）
        device = torch.device(args.device)
        # 直接创建用户指定的设备对象

    print(f"Device: {device}")
    # 打印当前使用的计算设备，方便用户确认
    print(f"GSA Chat Demo")
    # 打印项目标题横幅
    print("=" * 50)
    # 打印 50 个等号作为分隔线，增强可读性

    # 模型加载逻辑：根据参数选择加载训练好的模型或创建未训练模型
    if args.checkpoint:
        # 如果用户提供了 --checkpoint 路径
        model, step, loss = load_model(args.checkpoint, device)
        # 从 checkpoint 加载训练好的模型和元信息
        print(f"Loaded checkpoint (step {step}, loss {loss})")
        # 打印加载信息：训练步数和损失值
    elif args.untrained:
        # 如果用户使用了 --untrained 标志
        model = create_untrained_model(device)
        # 创建随机权重的未训练模型
        print("Using UNTRAINED model (random weights - output will be random)")
        # 警告用户输出将是随机的
    else:
        # 既没有 checkpoint 也没有 --untrained 标志
        print("ERROR: 需要 --checkpoint 或 --untrained")
        # 打印错误提示，告知用户必须提供其中一个参数
        sys.exit(1)
        # 以非零退出码退出程序（1 表示异常/错误），向操作系统报告失败

    n_params = model.count_parameters()
    # 调用模型的参数计数方法，获取模型总参数量（可训练参数）
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")
    # 打印模型参数量
    # {n_params:,}: 使用千分位分隔符格式化数字（如 12,345,678）
    # {n_params/1e6:.1f}M: 换算为百万 (M) 单位并保留一位小数（如 12.3M）
    print(f"Context window: {model.cfg.ctx_len} tokens")
    # 打印模型的上下文窗口大小（最大可处理的序列长度）
    print()
    # 打印空行，用于分隔输出内容

    # 加载分词器
    tokenizer, tok_type = get_tokenizer(args.tokenizer)
    # 调用 get_tokenizer，传入用户指定的 tokenizer 路径（可能为 None）
    if hasattr(tokenizer, 'get_vocab_size'):
        # 检查分词器是否有 get_vocab_size 方法（HuggingFace tokenizers 标准接口）
        vocab = tokenizer.get_vocab_size()
        # 获取词表实际大小
    else:
        # 否则（如 SimpleTokenizer 没有 get_vocab_size 方法）
        vocab = getattr(tokenizer, 'vocab_size', '?')
        # 尝试获取 vocab_size 属性，不存在则显示 "?"（未知）
    print(f"Tokenizer: {tok_type} (vocab={vocab})")
    # 打印分词器类型（BPE 或 simple）和词表大小
    print()
    # 空行分隔

    if args.interactive:
        # 如果启用了交互模式（--interactive 标志）
        print("交互模式 - 输入 prompt 生成回复，输入 /quit 退出")
        # 打印交互模式使用说明
        print("-" * 50)
        # 打印分隔线
        while True:
            # 无限循环，持续接收用户输入
            try:
                # 尝试读取用户输入
                prompt = input("\nYou: ").strip()
                # 显示 "You: " 提示符，读取用户输入并去除首尾空白字符
            except (EOFError, KeyboardInterrupt):
                # 捕获两种退出信号
                # EOFError: 用户按下 Ctrl+D (Unix) 或 Ctrl+Z+Enter (Windows)，表示输入结束
                # KeyboardInterrupt: 用户按下 Ctrl+C，表示强制中断程序
                print("\n再见！")
                # 打印告别信息
                break
                # 跳出 while 循环，程序结束

            if prompt.lower() in ('/quit', '/exit', '/q'):
                # 检查用户是否输入了退出命令（不区分大小写）
                print("再见！")
                # 打印告别信息
                break
                # 跳出循环
            if not prompt:
                # 如果用户输入为空（直接按回车）
                continue
                # 跳过本次循环，不生成回复，等待下一次有效输入

            print("GSA: ", end="", flush=True)
            # 打印 "GSA: " 前缀
            # end="": 不自动添加换行符，让生成的回复紧接着前缀输出
            # flush=True: 立即刷新输出缓冲区，确保在生成过程中 "GSA: " 就能显示在终端上
            response = generate_response(
                # 调用 generate_response 生成模型回复
                model, tokenizer, prompt, device,
                # 传递模型、分词器、用户输入、设备
                max_tokens=args.max_tokens,
                # 使用命令行指定的最大 token 数
                temperature=args.temperature,
                # 使用命令行指定的温度参数
            )
            print(response)
            # 打印模型生成的回复文本（自动换行）
    else:
        # 如果不是交互模式，执行单次 prompt 模式
        print(f"Prompt: {args.prompt}")
        # 打印用户指定的起始 prompt 文本
        print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
        # 打印生成参数（最大 token 数和温度）
        print("-" * 50)
        # 打印分隔线
        print("GSA: ", end="", flush=True)
        # 打印 "GSA: " 前缀并立即刷新缓冲区
        response = generate_response(
            # 调用 generate_response 生成回复
            model, tokenizer, args.prompt, device,
            # 使用命令行中指定的 prompt 参数
            max_tokens=args.max_tokens,
            # 最大生成 token 数
            temperature=args.temperature,
            # 采样温度
        )
        print(response)
        # 打印生成的回复文本（自动换行）


if __name__ == "__main__":
    # Python 的标准入口守卫：只有当此文件被直接运行时（而非被 import 导入时）才执行以下代码
    main()
    # 调用 main 函数，启动整个程序
    # 设计决策：这种模式使得 demo_chat.py 既可以作为独立脚本运行，也可以被其他模块导入（此时不自动执行 main）
