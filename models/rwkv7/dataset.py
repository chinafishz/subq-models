"""  # 模块文档字符串开始
Dataset Loader for RWKV-Courage  # 数据集加载器，为 RWKV-Courage 项目提供数据加载功能
================================  # 分隔线，标记标题结束

Handles loading, mixing, and tokenizing training data.  # 负责加载、混合和分词训练数据

Data sources (Phase 1):  # 第一阶段的数据来源说明
  - TinyStories (85%): ~500M tokens of synthetic children's stories  # TinyStories 数据集占85%，约5亿token的合成儿童故事
  - Digimon Courage Material (15%): Philosophy and narrative text  # 数码宝贝勇气材料占15%，包含哲学和叙事文本

Output format: tokenized arrays saved to .bin (binary tokens)  # 输出格式：分词后的数组保存为 .bin 二进制文件
"""  # 模块文档字符串结束

import os  # 导入操作系统接口模块，用于文件路径操作
import torch  # 导入 PyTorch 深度学习框架，用于张量操作
import numpy as np  # 导入 NumPy 数值计算库，别名为 np，用于数组操作
from typing import Iterator, Optional  # 从 typing 模块导入迭代器和可选类型注解
from datasets import load_dataset  # 从 HuggingFace datasets 库导入数据集加载函数
from tokenizers import Tokenizer  # 从 tokenizers 库导入分词器类


class MixedDataset:  # 定义混合数据集类，用于预训练的数据流式加载和混合
    """  # 类文档字符串开始
    Streams and mixes multiple text datasets for pre-training.  # 流式读取并混合多个文本数据集用于预训练
    
    Data is tokenized on-the-fly and yielded as (B, T) tensors.  # 数据即时分词并以 (批次大小, 序列长度) 张量形式产出
    """  # 类文档字符串结束

    def __init__(  # 构造函数，初始化 MixedDataset 实例
        self,  # 实例自身的引用
        tokenizer: Tokenizer,  # 分词器对象，用于将文本转换为 token ID
        tinystories_ratio: float = 0.85,  # TinyStories 数据的混合比例，默认 85%
        courage_ratio: float = 0.15,  # 勇气材料的混合比例，默认 15%
        seq_len: int = 4096,  # 序列长度，即每个训练样本的 token 数量，默认 4096
    ):  # 构造函数参数列表结束
        self.tokenizer = tokenizer  # 保存分词器到实例属性
        self.tinystories_ratio = tinystories_ratio  # 保存 TinyStories 混合比例
        self.courage_ratio = courage_ratio  # 保存勇气材料混合比例
        self.seq_len = seq_len  # 保存序列长度配置
        self.vocab_size = tokenizer.get_vocab_size()  # 获取并保存分词器的词汇表大小
        self.pad_token_id = tokenizer.token_to_id("[PAD]") or 0  # 获取填充 token 的 ID，若不存在则用 0

        # Lazy initialization  # 延迟初始化注释：流对象将在首次使用时创建
        self._tinystories_stream = None  # 初始化 TinyStories 数据流为 None（延迟加载）
        self._courage_stream = None  # 初始化勇气材料数据流为 None（延迟加载）

    def _load_tinystories(self) -> Iterator[str]:  # 私有方法：流式加载 TinyStories 数据集，返回字符串迭代器
        """Stream TinyStories dataset."""  # 方法文档字符串：流式读取 TinyStories 数据集
        ds = load_dataset(  # 调用 HuggingFace datasets 的 load_dataset 函数加载数据集
            "roneneldan/TinyStories",  # 数据集名称：roneneldan 发布的 TinyStories
            split="train",  # 加载训练集分割
            streaming=True,  # 启用流式加载模式，不一次性加载全部数据到内存
            trust_remote_code=False,  # 不信任远程代码，安全起见不执行远程脚本
        )  # load_dataset 调用结束
        for item in ds:  # 遍历数据集中的每个样本
            yield item["text"]  # 产出每个样本的 "text" 字段（故事文本）

    def _load_courage_material(self, courage_path: Optional[str]):  # 私有方法：加载勇气训练材料，参数为可选的文件路径
        """Load courage training material from text file."""  # 方法文档字符串：从文本文件加载勇气训练材料
        if courage_path is None:  # 如果文件路径为 None（未提供）
            return  # 直接返回，不加载任何数据
        if not os.path.exists(courage_path):  # 检查文件路径是否存在
            print(f"WARNING: Courage material not found at {courage_path}")  # 打印警告：未找到勇气材料文件
            print("  Using TinyStories-only mode until courage material is prepared.")  # 提示当前仅使用 TinyStories 模式
            return  # 返回，不加载数据
        with open(courage_path, "r", encoding="utf-8") as f:  # 以 UTF-8 编码打开勇气材料文本文件
            text = f.read()  # 读取整个文件的文本内容

        # Split into chunks of roughly 512-2048 chars  # 注释：将文本分割成约 512-2048 字符的块
        # (will be further tokenized by the training pipeline)  # 注释：后续会由训练管线进一步分词
        paragraphs = text.split("\n\n")  # 按双换行符分割文本为段落列表
        for para in paragraphs:  # 遍历每个段落
            para = para.strip()  # 去除段落首尾的空白字符
            if len(para) > 50:  # 如果段落长度大于 50 个字符（跳过过短的行）
                yield para  # 产出该段落文本

    def stream_tokens(self, courage_path: Optional[str] = None) -> Iterator[torch.Tensor]:  # 公有方法：流式产出分词后的序列，返回 PyTorch 张量迭代器
        """  # 方法文档字符串开始
        Yield tokenized sequences of length seq_len.  # 产出长度为 seq_len 的分词序列
        
        Mixes TinyStories and courage material according to configured ratios.  # 按配置比例混合 TinyStories 和勇气材料
        """  # 方法文档字符串结束
        tinystories = self._load_tinystories()  # 创建 TinyStories 数据流的迭代器
        courage = self._load_courage_material(courage_path) if courage_path else None  # 如果提供了勇气材料路径则创建迭代器，否则为 None

        buffer = []  # 初始化 token 缓冲区（列表），用于累积 token
        buffer_len = 0  # 初始化缓冲区当前 token 数量计数器

        while True:  # 无限循环，持续产出训练序列
            # Decide which source to pull from  # 注释：决定从哪个数据源拉取数据
            if courage and np.random.random() < self.courage_ratio:  # 如果勇气迭代器存在且随机数小于勇气比例（15%概率）
                try:  # 尝试从勇气迭代器获取下一个文本
                    text = next(courage)  # 从勇气材料迭代器获取下一个文本段落
                except StopIteration:  # 如果勇气材料迭代器耗尽
                    courage = self._load_courage_material(courage_path)  # 重新加载勇气材料迭代器
                    try:  # 尝试再次获取
                        text = next(courage)  # 从重新加载的勇气迭代器获取文本
                    except (StopIteration, TypeError):  # 如果仍然耗尽或迭代器无效
                        text = next(tinystories)  # 回退到 TinyStories 数据源获取文本
            else:  # 否则（不使用勇气材料，即 85% 概率）
                try:  # 尝试从 TinyStories 迭代器获取下一个文本
                    text = next(tinystories)  # 从 TinyStories 迭代器获取下一个故事文本
                except StopIteration:  # 如果 TinyStories 迭代器耗尽
                    tinystories = self._load_tinystories()  # 重新加载 TinyStories 迭代器（循环使用）
                    text = next(tinystories)  # 从重新加载的迭代器获取文本

            # Tokenize  # 注释：对文本进行分词
            encoded = self.tokenizer.encode(text)  # 使用分词器将文本编码为 token 对象
            tokens = encoded.ids  # 从编码结果中提取 token ID 列表

            buffer.extend(tokens)  # 将新 token 追加到缓冲区末尾
            buffer_len += len(tokens)  # 更新缓冲区 token 计数

            # Yield full sequences  # 注释：当缓冲区足够时产出完整序列
            while buffer_len >= self.seq_len + 1:  # 当缓冲区 token 数大于等于 seq_len+1 时（+1 是因为需要目标 token）
                seq = torch.tensor(buffer[:self.seq_len + 1], dtype=torch.long)  # 从缓冲区取 seq_len+1 个 token 构造长整型张量
                buffer = buffer[self.seq_len:]  # 滑动窗口：从缓冲区移除前 seq_len 个 token，保留剩余部分
                buffer_len = len(buffer)  # 更新缓冲区 token 计数
                yield seq  # 产出该序列张量给训练循环

    def prepare_binidx(self, output_dir: str, num_tokens: int,  # 公有方法：准备 .bin/.idx 格式文件（RWKV 原生格式），参数：输出目录和目标 token 数
                       courage_path: Optional[str] = None):  # 可选参数：勇气材料文件路径
        """  # 方法文档字符串开始
        Convert streaming data to .bin/.idx format (RWKV native format).  # 将流式数据转换为 .bin/.idx 格式（RWKV 原生格式）
        
        Args:  # 参数说明部分
            output_dir: directory for output files  # output_dir：输出文件的目录
            num_tokens: target number of tokens to process  # num_tokens：要处理的目标 token 数量
            courage_path: path to courage material text file  # courage_path：勇气材料文本文件的路径
        """  # 方法文档字符串结束
        os.makedirs(output_dir, exist_ok=True)  # 创建输出目录（如果已存在则不报错）
        bin_path = os.path.join(output_dir, "train.bin")  # 构造二进制 token 文件路径
        idx_path = os.path.join(output_dir, "train.idx")  # 构造索引文件路径

        # First pass: write tokens to .bin  # 注释：第一遍处理：将 token 写入 .bin 文件
        all_tokens = []  # 初始化用于存储所有 token 的列表
        stream = self.stream_tokens(courage_path)  # 创建混合数据流的 token 迭代器

        pbar = None  # 初始化进度条变量为 None
        try:  # 尝试导入 tqdm 进度条库
            from tqdm import tqdm  # 从 tqdm 库导入 tqdm 进度条类
            pbar = tqdm(total=num_tokens, desc="Tokenizing", unit="tok")  # 创建进度条，显示分词进度
        except ImportError:  # 如果 tqdm 未安装
            pass  # 跳过，不使用进度条

        total = 0  # 初始化已处理 token 总数计数器
        for seq in stream:  # 遍历流式产出的每个序列张量
            tokens = seq.tolist()  # 将 PyTorch 张量转换为 Python 列表
            all_tokens.extend(tokens)  # 将所有 token 追加到总列表
            total += len(tokens)  # 累加已处理 token 数
            if pbar:  # 如果进度条存在
                pbar.update(len(tokens))  # 更新进度条，显示本次新增的 token 数
            if total >= num_tokens:  # 如果已达到目标 token 数量
                break  # 跳出循环

        if pbar:  # 如果进度条存在
            pbar.close()  # 关闭进度条

        all_tokens = all_tokens[:num_tokens]  # 截取前 num_tokens 个 token（确保精确数量）

        # Write .bin (uint16 for vocab < 65536)  # 注释：写入 .bin 文件（词汇量小于 65536 时使用 uint16 类型）
        tokens_array = np.array(all_tokens, dtype=np.uint16)  # 将 token 列表转换为 uint16 类型的 NumPy 数组
        tokens_array.tofile(bin_path)  # 将数组直接写入二进制文件

        # Write .idx (text index for RWKV training)  # 注释：写入 .idx 文件（RWKV 训练的文本索引）
        with open(idx_path, "w") as f:  # 以写入模式打开索引文件
            f.write(f"{bin_path}\n")  # 将 .bin 文件路径写入索引文件

        print(f"Prepared {len(all_tokens):,} tokens -> {bin_path}")  # 打印完成信息：已准备的 token 数量和输出文件路径
        return bin_path, idx_path  # 返回 .bin 和 .idx 文件的路径