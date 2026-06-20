#!/usr/bin/env python3  # Shebang 声明：指定脚本使用 Python 3 解释器执行（Unix/Linux 环境下直接 ./train_gsa.py 可运行）
"""
GSA 模型训练脚本 — 包含分词器（Tokenizer）构建、TinyStories 数据加载、以及完整的训练循环。

GSA（Gated State Attention，门控状态注意力）是一种轻量级语言模型架构，
本脚本实现了从零开始训练 GSA 模型的完整流程：构建 BPE 分词器 → 加载训练数据 → 训练循环。

使用示例：
    # 完整训练流水线（训练分词器 + 模型），使用 MPS（Apple Silicon GPU）加速
    python train_gsa.py --steps 2000 --batch 4 --device mps

    # 从检查点恢复训练，继续训练到 5000 步
    python train_gsa.py --resume checkpoints/latest.pt --steps 5000
"""

import argparse  # Python 标准库：命令行参数解析，用于处理 --steps、--batch、--device 等训练参数
import os  # Python 标准库：操作系统接口，用于文件路径操作（os.path.exists、os.makedirs 等）
import sys  # Python 标准库：系统相关功能，用于动态修改 Python 模块搜索路径（sys.path.insert）
import time  # Python 标准库：时间相关，用于记录训练耗时（time.time() 获取时间戳）
import math  # Python 标准库：数学函数，用于余弦学习率衰减中的 math.cos、math.pi 计算
from pathlib import Path  # Python 标准库：面向对象的文件系统路径操作，替代字符串路径拼接

import torch  # PyTorch 深度学习框架：张量运算、自动微分、模型训练的核心依赖
import torch.nn.functional as F  # PyTorch 函数式 API：提供 F.cross_entropy 等损失函数（无需实例化 Module）

# ═══════════════════════════════════════════════════════════════════════════════
# 路径配置：确定脚本自身位置和项目根目录
# ═══════════════════════════════════════════════════════════════════════════════
SPIKE_DIR = Path(__file__).parent  # 当前脚本所在目录的绝对路径：models/gsa/（spike 意为"尖峰"项目目录）
PROJECT_DIR = SPIKE_DIR.parent.parent  # 项目根目录：向上两层，从 models/gsa/ → 项目根目录
sys.path.insert(0, str(SPIKE_DIR))  # 将当前脚本目录插入 Python 模块搜索路径最前面，确保能直接 import 同目录下的 model_gsa 模块

from model_gsa import GSA_Config, GSALanguageModel  # 从同目录下的 model_gsa.py 导入 GSA 配置类和语言模型类

# ═══════════════════════════════════════════════════════════════════════════════
# 第一部分：分词器（Tokenizer）构建
# ═══════════════════════════════════════════════════════════════════════════════

def build_tokenizer(input_texts: list, vocab_size: int = 2000, output_path: str = None):
    """
    在输入文本上训练 BPE（Byte Pair Encoding，字节对编码）分词器。
    如果 tokenizers 库不可用，则回退到简单的字符级分词器。

    参数:
        input_texts (list): 用于训练分词器的文本列表，每个元素为一段字符串
        vocab_size (int): 词表大小，默认 2000（小型实验用，生产环境通常用 32000+）
        output_path (str): 分词器模型文件保存路径（如 "data/gsa_tokenizer.json"），为 None 则不保存

    返回:
        tuple: (tokenizer对象, vocab_size实际大小)
            - tokenizer: HuggingFace tokenizers.Tokenizer 或 SimpleTokenizer 实例
            - vocab_size: 最终词表大小（可能与输入参数不同，由 tokenizer 实际决定）
    """
    try:
        # 尝试导入 HuggingFace tokenizers 库（高性能 Rust 实现的 BPE 分词器）
        # models: BPE 模型类；pre_tokenizers: 预分词器（ByteLevel）；decoders: 解码器
        # trainers: 训练器（BpeTrainer）；processors: 后处理器（归一化 token 序列）
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

        # ── 构建 BPE Tokenizer 对象 ──
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))  # 创建 BPE 模型的分词器，未登录词用 [UNK] 替代
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # 设置 ByteLevel 预分词器：将文本转为字节级表示，不添加前导空格（适合英文）
        tok.decoder = decoders.ByteLevel()  # 设置 ByteLevel 解码器：将 token ID 序列还原为原始文本
        tok.post_processor = processors.ByteLevel(trim_offsets=False)  # 设置后处理器：保持原始偏移量，不裁剪（便于对齐原始文本位置）

        # ── 配置 BPE 训练器 ──
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,  # 目标词表大小：控制最终 token 数量上限
            min_frequency=2,  # 最小合并频率：只有出现 ≥2 次的字节对才会被合并入词表（过滤低频噪声）
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],  # 特殊 token 列表：[UNK]=未知词, [PAD]=填充, [BOS]=句子开头, [EOS]=句子结尾
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 初始字母表：ByteLevel 的 256 个字节字符，作为 BPE 的初始基础符号
        )

        # ── 将文本写入临时文件供 tokenizers 训练使用 ──
        # 注意：HuggingFace tokenizers 的 train() 方法要求传入文件路径列表
        import tempfile  # Python 标准库：创建临时文件和目录
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:  # 创建临时文本文件（写入模式，UTF-8 编码，关闭后不自动删除）
            for text in input_texts:  # 遍历所有训练文本
                f.write(text + '\n')  # 每行写入一段文本，换行符分隔
            tmp_path = f.name  # 获取临时文件的完整路径

        tok.train(files=[tmp_path], trainer=trainer)  # 在临时文件上训练 BPE 分词器：多次迭代合并高频字节对
        os.unlink(tmp_path)  # 训练完成后删除临时文件，避免磁盘残留

        # ── 可选：保存分词器到磁盘 ──
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)  # 递归创建输出目录（如 data/），若已存在则跳过
            tok.save(output_path)  # 将训练好的分词器保存为 JSON 文件（包含词表、合并规则等）

        print(f"BPE tokenizer: vocab={tok.get_vocab_size()} (saved to {output_path})")  # 打印日志：BPE 分词器构建完成，显示实际词表大小
        return tok, tok.get_vocab_size()  # 返回分词器对象和实际词表大小

    except ImportError:
        # tokenizers 库未安装时的回退方案：使用简单字符级分词器
        print("tokenizers not installed — using simple char-level tokenizer")  # 警告用户：tokenizers 库不可用，降级为字符级分词
        return SimpleTokenizer(vocab_size), vocab_size  # 创建 SimpleTokenizer 实例并返回


class SimpleTokenizer:
    """
    简单字符级分词器（回退方案）。
    
    当 HuggingFace tokenizers 库不可用时使用。原理：将每个字符映射到固定 ID，
    ID = 4 + (字符 Unicode 码点 % (词表大小 - 4))，这是一种有损但简单的编解码方式。
    
    特殊 token ID 固定分配：
        [PAD] = 0（填充，用于批处理中对齐不等长序列）
        [UNK] = 1（未知字符，理论上不会出现但保留位置）
        [BOS] = 2（句子开头，每个序列编码时自动在开头插入）
        [EOS] = 3（句子结尾，标识序列终止）
        用户字符从 ID=4 开始
    """
    def __init__(self, vocab_size: int = 2000):
        """
        初始化分词器。

        参数:
            vocab_size (int): 词表大小，默认 2000。决定字符到 ID 映射的取模范围。
        """
        self.vocab_size = vocab_size  # 保存词表大小，编码和解码时用于取模运算
        self.pad_id = 0  # [PAD] 的 token ID：填充符号，用于将不等长序列对齐到相同长度
        self.unk_id = 1  # [UNK] 的 token ID：未知符号，处理词表外字符（但在字符级分词中概率极低）
        self.bos_id = 2  # [BOS] 的 token ID：句子起始符号，每个序列编码时自动在开头插入
        self.eos_id = 3  # [EOS] 的 token ID：句子结束符号，标识序列终止位置

    def encode(self, text: str):
        """
        将文本编码为 token ID 序列。

        编码算法：
        1. 序列开头添加 BOS token（ID=2）
        2. 对每个字符 ch：tid = 4 + (ord(ch) % (vocab_size - 4))
           - ord(ch) 获取 Unicode 码点（如 'a' → 97）
           - 对 (vocab_size - 4) 取模，将码点映射到可用范围
           - +4 跳过前 4 个特殊 token 位置
        3. 用 min(tid, vocab_size - 1) 防止溢出

        参数:
            text (str): 待编码的文本字符串

        返回:
            Encoded 对象：包含 ids 属性的命名元组风格对象，可直接通过 .ids 访问 token ID 列表
        """
        ids = [self.bos_id]  # 初始化 ID 列表，首元素为 BOS token（句子开头标记）
        for ch in text:  # 遍历文本中的每一个字符
            tid = 4 + (ord(ch) % (self.vocab_size - 4))  # 核心映射：Unicode码点取模 → 偏移 → token ID（跳过特殊 token 占用的 0-3 位置）
            ids.append(min(tid, self.vocab_size - 1))  # 安全裁剪：确保 ID 不超过词表最大索引（vocab_size - 1），防止越界
        return type('Encoded', (), {'ids': ids})()  # 动态创建 Encoded 类型并实例化，模拟 HuggingFace 分词器返回格式，通过 .ids 访问

    def decode(self, ids: list) -> str:
        """
        将 token ID 序列解码回文本。

        解码算法（编码的逆过程）：
        1. 跳过 ID ≤ 3 的特殊 token（[PAD], [UNK], [BOS], [EOS]）
        2. 对每个有效 ID：chr(32 + (i - 4) % 95)
           - (i - 4) 反向减去特殊 token 偏移
           - % 95 取模映射到可打印 ASCII 范围（95 个可打印字符，从空格 32 开始）

        参数:
            ids (list): token ID 列表

        返回:
            str: 解码后的文本字符串（仅包含可打印 ASCII 字符）
        """
        chars = []  # 解码字符列表，逐步收集解码出的字符
        for i in ids:  # 遍历所有 token ID
            if i <= 3:  # 跳过特殊 token（ID 0-3：[PAD], [UNK], [BOS], [EOS]），它们不表示文本内容
                continue  # 不做任何处理，继续下一个 ID
            chars.append(chr(32 + (i - 4) % 95))  # 反向映射：减去特殊 token 偏移 → 取模 95 → 加 32 映射到可打印 ASCII 范围（空格到 '~'）
        return ''.join(chars)  # 将所有解码字符拼接成最终字符串

    def token_to_id(self, token: str):
        """
        将特殊 token 字符串映射为对应的 ID。

        参数:
            token (str): 特殊 token 名称（如 '[PAD]', '[UNK]', '[BOS]', '[EOS]'）

        返回:
            int 或 None: 对应的 token ID，若不在映射表中则返回 None
        """
        mapping = {'[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3}  # 特殊 token 到 ID 的硬编码映射表
        return mapping.get(token, None)  # 从映射表中查找，找不到返回 None

    def get_vocab_size(self):
        """
        获取词表大小。

        返回:
            int: 当前分词器的词表大小
        """
        return self.vocab_size  # 直接返回初始化时设置的词表大小

    def save(self, path: str):
        """
        将分词器配置保存为 JSON 文件。

        参数:
            path (str): 保存路径（如 "data/gsa_tokenizer.json"）
        """
        import json  # Python 标准库 JSON 模块（局部导入，仅在保存时使用）
        with open(path, 'w') as f:  # 以写入模式打开文件
            json.dump({'vocab_size': self.vocab_size}, f)  # 保存词表大小到 JSON（简单分词器只有 vocab_size 一个可配置参数）

    @staticmethod  # 静态方法：不需要实例化即可调用，用于从文件恢复分词器
    def from_file(path: str):
        """
        从 JSON 文件恢复 SimpleTokenizer 实例。

        参数:
            path (str): 分词器配置文件路径

        返回:
            SimpleTokenizer: 恢复后的分词器实例
        """
        import json  # 局部导入 JSON 模块
        with open(path) as f:  # 以只读模式打开文件
            d = json.load(f)  # 加载 JSON 数据到字典
        return SimpleTokenizer(d['vocab_size'])  # 用保存的 vocab_size 参数重新实例化分词器


# ═══════════════════════════════════════════════════════════════════════════════
# 第二部分：数据加载（TinyStories 数据集）
# ═══════════════════════════════════════════════════════════════════════════════

def get_tinystories_data(max_texts: int = 5000):
    """
    加载 TinyStories 数据集（一个小型英文故事语料库，适合轻量级语言模型训练）。
    
    TinyStories 是由 GPT-3.5/4 生成的短篇儿童故事数据集，包含约 270 万条故事。
    每个故事长度在 100-500 词之间，语言简单、语法规范，非常适合训练小型语言模型。

    优先从 HuggingFace 在线加载；如果网络不可用或 datasets 库未安装，则回退到内置示例文本。

    参数:
        max_texts (int): 最多加载的文本数量，默认 5000（控制训练数据规模）

    返回:
        list[str]: 文本字符串列表
    """
    texts = []  # 初始化空列表，用于存储加载的文本
    try:
        from datasets import load_dataset  # 尝试导入 HuggingFace datasets 库的 load_dataset 函数（在线数据加载）
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True,  # 以流式模式加载 TinyStories 训练集
                          trust_remote_code=False)  # 不信任远程代码（安全性考虑，避免执行数据集中的恶意 Python 代码）
        for i, item in enumerate(ds):  # 遍历流式数据集中的每一条记录
            if i >= max_texts:  # 达到最大文本数量限制
                break  # 退出循环，不再加载更多数据
            texts.append(item["text"])  # 提取每条记录的 "text" 字段（即故事内容）并添加到列表
        print(f"Loaded {len(texts)} TinyStories from HuggingFace")  # 日志：从 HuggingFace 成功加载的文本数量
        return texts  # 返回加载的文本列表
    except Exception as e:
        # 任何异常（网络不通、库未安装、权限问题等）都触发回退
        print(f"HuggingFace unavailable ({e}), using built-in samples")  # 警告：HuggingFace 不可用，使用内置样本
        return _get_sample_texts(max_texts)  # 调用回退函数获取内置示例文本


def _get_sample_texts(n: int):
    """
    回退方案：返回一组内置于脚本中的英文样本故事文本。
    
    当 HuggingFace 数据集不可用时，使用这些人工编写的样本作为训练数据。
    通过重复样本列表来达到所需的文本数量 n。

    参数:
        n (int): 需要返回的文本数量

    返回:
        list[str]: 文本列表（可能包含重复文本）
    """
    samples = [  # 内置的 25 条英文样本故事（硬编码于脚本中，用作离线训练数据）
        "Once upon a time, a brave little cat went to explore the forest.",  # "从前，一只勇敢的小猫去森林探险。"
        "I believe that courage is the first step to any great adventure.",  # "我相信勇气是任何伟大冒险的第一步。"
        "The sun set behind the mountains as the travelers made camp.",  # "太阳落在山后，旅行者们扎营了。"
        "She opened the old book and found a secret message inside.",  # "她打开旧书，发现里面有一条秘密信息。"
        "The robot learned to paint beautiful pictures of the stars.",  # "机器人学会了画星星的美丽图画。"
        "A kind dragon helped the villagers rebuild their homes after the storm.",  # "一条善良的龙帮助村民在暴风雨后重建家园。"
        "Deep in the ocean, a curious fish discovered a glowing crystal.",  # "在海洋深处，一条好奇的鱼发现了一颗发光的水晶。"
        "The young wizard practiced her spells every morning at dawn.",  # "年轻的女巫每天黎明时分练习她的咒语。"
        "They walked together through the garden, talking about their dreams.",  # "他们一起走过花园，谈论着各自的梦想。"
        "The little bird sang a song that made everyone in the forest smile.",  # "小鸟唱了一首歌，让森林里的每个人都笑了。"
        "He fixed the broken machine using only a screwdriver and some tape.",  # "他只用一把螺丝刀和一些胶带就修好了坏掉的机器。"
        "The moon was full and bright, casting silver light on the water.",  # "月亮又圆又亮，在水面上洒下银色的光芒。"
        "She decided to try again, even though she had failed many times.",  # "她决定再试一次，尽管已经失败了很多次。"
        "A mysterious package arrived at the doorstep on a rainy Tuesday.",  # "一个神秘的包裹在一个下雨的星期二送到了门口。"
        "The brave knight faced the dragon, not with a sword, but with kindness.",  # "勇敢的骑士面对巨龙，不是用剑，而是用善意。"
        "They built a treehouse high up in the old oak tree.",  # "他们在老橡树的高处建了一座树屋。"
        "The scientist made a discovery that would change everything.",  # "科学家做出了一项将改变一切的发现。"
        "Every night, the lighthouse keeper would light the great lamp.",  # "每个夜晚，灯塔看守人都会点亮那盏大灯。"
        "The fox taught the rabbit how to find the sweetest berries.",  # "狐狸教兔子如何找到最甜的浆果。"
        "I want to be a doctor so I can help people feel better.",  # "我想成为一名医生，这样我就可以帮助人们恢复健康。"
        "A shooting star streaked across the sky and they all made a wish.",  # "一颗流星划过天空，他们都许了愿。"
        "The old clock tower had been silent for a hundred years.",  # "古老的钟楼已经沉寂了一百年。"
        "She picked up the paintbrush and began to create a masterpiece.",  # "她拿起画笔，开始创作一幅杰作。"
        "The wind carried the seeds far across the meadow to new ground.",  # "风把种子带到了草原远方的新土地上。"
        "He wrote a letter to his future self and buried it under the apple tree.",  # "他给未来的自己写了一封信，埋在了苹果树下。"
    ]
    # ── 如果 n > 25 条样本，循环重复填充 ──
    result = []  # 最终返回的文本列表
    while len(result) < n:  # 当结果列表长度不足 n 时
        result.extend(samples)  # 将全部 25 条样本追加到结果列表（每次 extend 增加 25 条）
    return result[:n]  # 截取前 n 条返回（多余的部分被丢弃）


# ═══════════════════════════════════════════════════════════════════════════════
# 第三部分：核心训练循环
# ═══════════════════════════════════════════════════════════════════════════════

def train(config: dict):
    """
    GSA 语言模型的主训练函数。

    训练流程：
    1. 设备选择（MPS/CUDA/CPU）
    2. 数据加载与分词器构建
    3. 模型初始化（GSALanguageModel）
    4. 优化器配置（AdamW）
    5. 可选的检查点恢复
    6. 训练循环：批次构建 → 前向传播 → 损失计算 → 反向传播 → 梯度累积 → 优化器更新
    7. 最终模型保存

    参数:
        config (dict): 嵌套字典，包含两个子配置：
            - config["model"]: 模型架构参数（n_embd, n_layer, vocab_size, ctx_len 等）
            - config["training"]: 训练超参数（batch_size, learning_rate, max_steps 等）
    """
    cfg_model = config["model"]  # 提取模型配置字典（架构超参数）
    cfg_train = config["training"]  # 提取训练配置字典（优化器、步数等训练超参数）

    # ── 设备选择（GPU/MPS/CPU）──
    device_str = cfg_train.get("device", "cpu")  # 获取用户指定的设备字符串，默认使用 CPU
    if device_str == "mps" and torch.backends.mps.is_available():  # 用户选择 MPS（Metal Performance Shaders，Apple Silicon GPU 加速）且 MPS 后端可用
        device = torch.device("mps")  # 设置为 MPS 设备（适用于 M1/M2/M3/M4 等 Apple 芯片）
    elif device_str == "cuda" and torch.cuda.is_available():  # 用户选择 CUDA（NVIDIA GPU）且 CUDA 后端可用
        device = torch.device("cuda")  # 设置为 CUDA 设备（适用于 NVIDIA GPU）
    else:
        device = torch.device("cpu")  # 回退到 CPU（无 GPU 或用户指定了不支持的值时）

    print(f"Device: {device} | MPS: {torch.backends.mps.is_available()}")  # 打印最终使用的设备和 MPS 可用性状态

    # ── 数据准备 ──
    texts = get_tinystories_data(cfg_train.get("max_texts", 2000))  # 加载 TinyStories 文本数据，默认最多 2000 条
    tok_path = cfg_train.get("tokenizer_path", "data/gsa_tokenizer.json")  # 获取分词器保存路径，默认 "data/gsa_tokenizer.json"
    tokenizer, vocab_size = build_tokenizer(texts, cfg_model["vocab_size"], tok_path)  # 在训练文本上构建 BPE 分词器，返回分词器对象和实际词表大小

    cfg_model["vocab_size"] = vocab_size  # 用实际词表大小更新模型配置（分词器返回的 vocab_size 可能与原始配置不同）
    gsa_cfg = GSA_Config(cfg_model)  # 根据配置字典创建 GSA 模型配置对象（GSA_Config 是 dataclass，包含所有架构参数）

    # ── 预分词化所有训练文本（避免每次迭代重复编码） ──
    tokenized = []  # 存储所有文本的 token ID 张量（形状：[seq_len]）
    for text in texts:  # 遍历每条文本
        encoded = tokenizer.encode(text)  # 使用分词器将文本编码为 token ID 序列（返回对象有 .ids 属性）
        tokenized.append(torch.tensor(encoded.ids, dtype=torch.long))  # 将 token ID 列表转为 PyTorch 长整型张量（long=int64，用于 embedding 索引）

    print(f"Tokenized {len(tokenized)} texts, vocab={vocab_size}")  # 日志：完成分词，显示文本数量和词表大小

    # ── 模型初始化 ──
    model = GSALanguageModel(gsa_cfg).to(device)  # 创建 GSA 语言模型实例，并移动到指定设备（GPU/MPS/CPU）
    n_params = model.count_parameters()  # 计算模型总参数量（包括可训练和不可训练参数）
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")  # 打印参数量（千分位分隔，如 "2,500,000"），同时显示以百万为单位的近似值

    # ── 优化器配置（AdamW）──
    # AdamW 是 Adam 的改进版，将权重衰减（weight decay）与自适应学习率解耦，
    # 在 Transformer 训练中表现优于标准 Adam，是当前语言模型训练的首选优化器。
    lr = cfg_train.get("learning_rate", 3e-4)  # 基础学习率，默认 3e-4（0.0003），典型的小模型训练学习率
    optimizer = torch.optim.AdamW(
        model.parameters(),  # 模型的所有可训练参数
        lr=lr,  # 学习率
        betas=(cfg_train.get("beta1", 0.9), cfg_train.get("beta2", 0.95)),  # Adam 的动量参数：beta1=一阶矩衰减率（默认0.9），beta2=二阶矩衰减率（默认0.95，略高于常见的0.999以加快适应）
        weight_decay=cfg_train.get("weight_decay", 0.1),  # 权重衰减系数（L2 正则化），默认 0.1
    )

    # ── 检查点恢复（可选） ──
    start_step = 0  # 训练起始步数（默认从 0 开始）
    checkpoint_dir = cfg_train.get("checkpoint_dir", "checkpoints_gsa")  # 检查点保存目录
    resume_path = cfg_train.get("resume_from", None)  # 恢复路径：如果指定了检查点文件路径则从该点恢复
    if resume_path and os.path.exists(resume_path):  # 如果指定了恢复路径且文件确实存在
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)  # 加载检查点：map_location 确保张量加载到当前设备；weights_only=False 允许加载完整 Python 对象（包括优化器状态）
        model.load_state_dict(ckpt["model_state_dict"])  # 恢复模型权重（严格匹配 state_dict 的键名）
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # 恢复优化器状态（动量、方差等，确保优化器从断点继续）
        start_step = ckpt.get("step", 0)  # 获取保存时的训练步数，从该步继续（默认 0）
        print(f"Resumed from step {start_step}")  # 日志：成功恢复，显示起始步数

    # ── 训练超参数 ──
    batch_size = cfg_train.get("batch_size", 4)  # 每批次样本数，默认 4（小批量，适合显存受限场景）
    grad_accum = cfg_train.get("gradient_accumulation_steps", 8)  # 梯度累积步数：每 grad_accum 步才执行一次优化器更新（等效于增大批量的效果）
    max_steps = cfg_train.get("max_steps", 2000)  # 最大训练步数，默认 2000
    warmup_steps = cfg_train.get("warmup_steps", 100)  # 学习率预热步数：前 warmup_steps 步学习率从 0 线性增长到 lr
    min_lr = cfg_train.get("min_lr", 3e-5)  # 学习率下限：余弦衰减的最低学习率（通常是初始学习率的 1/10）
    ctx_len = min(cfg_model.get("ctx_len", 256), 256)  # 上下文长度（序列最大 token 数），默认 256，限制上限为 256 以控制训练速度
    grad_clip = cfg_train.get("grad_clip", 1.0)  # 梯度裁剪阈值：防止梯度爆炸，默认 1.0（梯度范数超过此值则缩放）

    # ── 训练状态初始化 ──
    model.train()  # 设置模型为训练模式（启用 Dropout、LayerNorm 的训练行为等）
    opt_step = start_step  # 优化器步数计数器（独立于训练步数，仅每次优化器更新时递增）
    optimizer.zero_grad()  # 清空所有参数的梯度缓存（训练前确保梯度从零开始）
    total_loss = 0.0  # 累积损失值（用于计算日志间隔内的平均损失）
    best_loss = float("inf")  # 最佳损失值（初始化为正无穷，用于跟踪训练过程中的最低损失）
    t0 = time.time()  # 记录当前时间戳，用于计算每个日志间隔的耗时

    # ── 训练日志：打印关键超参数 ──
    print(f"\nTraining: bs={batch_size}, accum={grad_accum}, "  # 批量大小 + 梯度累积步数
          f"eff_bs={batch_size*grad_accum}")  # 有效批量大小 = 实际 batch_size × 梯度累积步数（与 GPU 显存无关的逻辑批量大小）
    print(f"  ctx={ctx_len}, steps={start_step}→{max_steps}, lr={lr}\n")  # 上下文长度 + 训练步数范围 + 初始学习率

    # ═══════════════════════════════════════════════════════════════════════════
    # 主训练循环
    # ═══════════════════════════════════════════════════════════════════════════
    for step in range(start_step, max_steps):  # 从 start_step 到 max_steps-1 遍历每一步
        # ── 构建当前批次的输入/目标张量 ──
        # x_batch: 输入序列（前 n-1 个 token，模型用于预测后续 token）
        # y_batch: 目标序列（后 n 个 token，模型需要预测的正确 token）
        x_batch = torch.zeros(batch_size, ctx_len, dtype=torch.long, device=device)  # 创建全零输入张量：[batch_size, ctx_len]，长整型，放在指定设备上
        y_batch = torch.zeros(batch_size, ctx_len, dtype=torch.long, device=device)  # 创建全零目标张量：[batch_size, ctx_len]，长整型，放在指定设备上

        for b in range(batch_size):  # 遍历批次中的每个样本
            # ── 随机选择一条文本，随机截取一个窗口 ──
            idx_text = torch.randint(0, len(tokenized), (1,)).item()  # 从所有已分词文本中随机选取一个索引（均匀分布）
            tokens = tokenized[idx_text]  # 获取该文本的 token 张量（形状：[text_len]）
            if len(tokens) <= ctx_len + 1:  # 如果文本长度 ≤ ctx_len+1（文本太短，无法填充整个上下文窗口）
                # 短文本处理：用零填充（0 即 [PAD] token ID）
                x_batch[b, :len(tokens)-1] = tokens[:-1]  # 输入：取前 len-1 个 token（不包括最后一个）
                y_batch[b, :len(tokens)-1] = tokens[1:]  # 目标：取后 len-1 个 token（shifted by 1，即预测下一个 token）
                y_batch[b, len(tokens)-1:] = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0  # 超出文本长度的目标位置填充为 [PAD]（0），这些位置不参与损失计算
            else:  # 文本足够长，可以随机截取
                start = torch.randint(0, len(tokens) - ctx_len - 1, (1,)).item()  # 随机选择起始位置（范围：0 到 len - ctx_len - 1，确保能完整取 ctx_len+1 个 token）
                x_batch[b] = tokens[start:start + ctx_len]  # 输入：从 start 位置取 ctx_len 个连续 token
                y_batch[b] = tokens[start + 1:start + ctx_len + 1]  # 目标：从 start+1 位置取 ctx_len 个连续 token（shifted by 1，即 Next Token Prediction 任务）

        # ── 前向传播 ──
        logits = model(x_batch)  # 模型前向计算：输入 x_batch [B, L]，输出 logits [B, L, V]（V=vocab_size，每个位置的词表概率分布）
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),  # 将 logits 展平为 [B*L, V]：二维交叉熵所需格式（预测分布）
            y_batch.view(-1),  # 将目标展平为 [B*L]：一维类别标签（真实 token ID）
            ignore_index=-100,  # 忽略值为 -100 的位置（短文本填充位置的目标为 0=[PAD]，不会被忽略；如需忽略填充，应设为 -100 或 tokenizer.pad_id）
        )

        # ── 数值稳定性检查：检测 NaN/Inf 损失 ──
        if not torch.isfinite(loss):  # 如果损失不是有限值（NaN 或 Inf）
            print(f"ERROR: NaN loss at step {step + 1}. Aborting.")  # 报错并显示步数（step+1 转为人类可读的 1-indexed）
            break  # 立即终止训练循环（梯度爆炸或数值问题，继续训练无意义）

        # ── 梯度累积：缩放损失后反向传播 ──
        loss = loss / grad_accum  # 损失除以梯度累积步数：使得多次小批量的梯度累加等效于一次大批量的效果
        loss.backward()  # 反向传播：计算所有参数的梯度并累加到 .grad 属性中

        # ── 优化器步进（仅在累积足够步数后执行） ──
        if (step + 1) % grad_accum == 0:  # 当训练步数对梯度累积步数取模为 0 时（如每 8 步执行一次）
            opt_step += 1  # 优化器步数递增

            # ── 学习率调度（余弦衰减 + 预热） ──
            if opt_step < warmup_steps:  # 预热阶段：前 warmup_steps 步
                current_lr = lr * opt_step / max(1, warmup_steps)  # 线性预热：学习率从 0 线性增长到 lr（max(1,...) 防止除零）
            else:  # 余弦衰减阶段：warmup_steps 之后
                progress = (opt_step - warmup_steps) / max(1, max_steps // grad_accum - warmup_steps)  # 计算训练进度比例（0→1）：当前步数在衰减区间的相对位置
                current_lr = min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))  # 余弦退火公式：从 lr 平滑降至 min_lr，cos 在 [0,π] 上从 1 到 -1，乘以 0.5 缩放到 [0,1]
                                                                                                   # 完整公式推导：lr(t) = min_lr + 0.5*(lr - min_lr)*(1 + cos(π*t/T))
                                                                                                   # t=0 时 cos(0)=1 → lr(t)=lr; t=T 时 cos(π)=-1 → lr(t)=min_lr

            for pg in optimizer.param_groups:  # 遍历优化器的所有参数组（通常只有一个组）
                pg["lr"] = current_lr  # 将当前计算的学习率赋值给参数组

            # ── 梯度裁剪（防止梯度爆炸） ──
            if grad_clip > 0:  # 仅当梯度裁剪阈值大于 0 时才执行（0 或负数表示不裁剪）
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # 按全局梯度范数裁剪：若 |g| > grad_clip，则将 g 缩放为 g * grad_clip / |g|
            optimizer.step()  # 执行参数更新：根据 AdamW 算法更新所有权重
            optimizer.zero_grad()  # 清空梯度缓存，为下一轮梯度累积做准备

        total_loss += loss.item() * grad_accum  # 累加未经缩放的原始损失（loss.item() 是将标量张量转为 Python float，乘 grad_accum 还原为真实损失值）

        # ── 日志输出 ──
        log_interval = cfg_train.get("log_interval", 50)  # 日志打印间隔（步数），默认每 50 步输出一次
        if (step + 1) % log_interval == 0:  # 达到日志间隔
            avg_loss = total_loss / log_interval  # 计算该间隔内的平均损失
            elapsed = time.time() - t0  # 计算该间隔的耗时（秒）
            print(f"step {step+1:5d}/{max_steps} | loss {avg_loss:.4f} | "  # 打印步数（5 位右对齐）、训练总步数和平均损失（4 位小数）
                  f"lr {current_lr:.2e} | {elapsed:.1f}s")  # 打印当前学习率（科学记数法 2 位小数）和间隔耗时
            total_loss = 0.0  # 重置累积损失，为下一个日志间隔做准备
            t0 = time.time()  # 重置计时起点

            if avg_loss < best_loss:  # 如果当前平均损失低于历史最佳损失
                best_loss = avg_loss  # 更新最佳损失值

    # ── 训练结束：保存最终模型 ──
    os.makedirs(checkpoint_dir, exist_ok=True)  # 创建检查点目录（若已存在则跳过）
    final_path = os.path.join(checkpoint_dir, "final.pt")  # 最终模型保存路径：checkpoints_gsa/final.pt
    torch.save({
        "step": step + 1,  # 保存当前的训练步数（1-indexed，用于后续恢复）
        "model_state_dict": model.state_dict(),  # 模型权重（OrderedDict，包含所有参数名和对应的张量）
        "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态（动量、方差、步数等，保证完整恢复训练状态）
        "config": {k: v for k, v in gsa_cfg.__dict__.items() if not k.startswith("_")},  # 模型配置（过滤掉以下划线开头的私有属性）
    }, final_path)  # 保存到最终路径
    print(f"\nModel saved: {final_path}")  # 日志：模型保存路径
    print(f"Best loss: {best_loss:.4f}")  # 日志：训练过程中的最佳损失值


# ═══════════════════════════════════════════════════════════════════════════════
# 第四部分：命令行接口（CLI）
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    命令行入口函数：解析参数 → 构建配置字典 → 调用 train() 开始训练。
    
    支持的参数：
        --steps: 最大训练步数
        --batch: 批次大小
        --ctx: 上下文长度（序列最大 token 数）
        --device: 训练设备（cpu/mps/cuda）
        --resume: 从检查点恢复训练的路径
        --tokenizer: 分词器保存/加载路径
    """
    parser = argparse.ArgumentParser(description="Train GSA Language Model")  # 创建参数解析器，设置程序描述
    parser.add_argument("--steps", type=int, default=2000)  # --steps 参数：最大训练步数，整数类型，默认 2000
    parser.add_argument("--batch", type=int, default=4)  # --batch 参数：批次大小，整数类型，默认 4
    parser.add_argument("--ctx", type=int, default=256)  # --ctx 参数：上下文窗口长度，整数类型，默认 256
    parser.add_argument("--device", default="cpu")  # --device 参数：训练设备，字符串类型，默认 "cpu"
    parser.add_argument("--resume", default=None)  # --resume 参数：检查点恢复路径，字符串类型，默认 None（不恢复）
    parser.add_argument("--tokenizer", default="data/gsa_tokenizer.json")  # --tokenizer 参数：分词器文件路径，默认 "data/gsa_tokenizer.json"
    args = parser.parse_args()  # 解析命令行参数，返回包含所有参数的命名空间对象

    # ── 构建配置字典（嵌套结构：model + training） ──
    config = {
        "model": {  # 模型架构配置
            "n_embd": 384,  # 嵌入维度（hidden size）：token embedding 和隐藏状态的维度，默认 384
            "n_layer": 8,  # Transformer 层数（GSA 块的数量），默认 8 层
            "vocab_size": 2000,  # 词表大小（实际会被 build_tokenizer 返回的值覆盖），初始默认 2000
            "ctx_len": args.ctx,  # 上下文长度（最大序列长度），由命令行参数 --ctx 指定
            "n_head": 6,  # 注意力头数：多头注意力的头数量，默认 6 头
            "head_dim": 64,  # 每个注意力头的维度：总嵌入维度 = n_head × head_dim = 6 × 64 = 384
            "global_k": 64,  # 全局注意力 Key/Value 的压缩维度：线性注意力中 K/V 投影到更低维度进行高效计算
            "window": 32,  # 局部窗口大小：滑动窗口注意力的窗口长度，每个 token 只关注前后 window 范围内的 token
        },
        "training": {  # 训练超参数配置
            "batch_size": args.batch,  # 每批次样本数，由 --batch 指定
            "gradient_accumulation_steps": 4,  # 梯度累积步数：每 4 步更新一次参数（等效批量 = batch_size × 4）
            "learning_rate": 3e-4,  # 初始学习率（AdamW），3e-4 = 0.0003
            "min_lr": 3e-5,  # 最低学习率（余弦衰减终点）：初始 lr 的 1/10
            "warmup_steps": 50,  # 学习率预热步数：前 50 步线性增长
            "max_steps": args.steps,  # 最大训练步数，由 --steps 指定
            "max_texts": max(500, args.steps * 2),  # 最大加载文本数：至少 500 条，或训练步数的 2 倍（确保数据多样性）
            "weight_decay": 0.1,  # 权重衰减系数（AdamW 的 L2 正则化强度）
            "beta1": 0.9,  # Adam 一阶矩衰减率（动量参数）
            "beta2": 0.95,  # Adam 二阶矩衰减率（方差参数），略低于常见 0.999 以更快适应
            "grad_clip": 1.0,  # 梯度裁剪阈值：梯度 L2 范数超过 1.0 则缩放
            "log_interval": 20,  # 日志打印间隔（步数）：每 20 步输出一次训练状态
            "save_interval": 500,  # 模型保存间隔（步数）：每 500 步保存一次检查点（当前版本未使用此参数）
            "device": args.device,  # 训练设备（cpu/mps/cuda），由 --device 指定
            "tokenizer_path": args.tokenizer,  # 分词器文件路径，由 --tokenizer 指定
            "checkpoint_dir": "checkpoints_gsa",  # 检查点保存目录
            "resume_from": args.resume,  # 恢复训练的检查点路径，由 --resume 指定
        },
    }

    train(config)  # 调用核心训练函数，传入完整配置字典


# ── 脚本入口：仅当直接运行此文件时执行 main()，被 import 时不执行 ──
if __name__ == "__main__":  # Python 惯用模式：判断当前模块是否作为主程序运行
    main()  # 调用命令行入口函数，开始训练流程
