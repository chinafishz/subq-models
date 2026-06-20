"""  # 开始模块文档字符串
BPE Tokenizer Trainer for RWKV-Courage  # 模块标题：RWKV-Courage 项目的 BPE 分词器训练器
=======================================  # 标题分隔线
Trains a Byte-Pair Encoding (BPE) tokenizer from scratch on the training corpus  # 在训练语料库上从头训练一个字节对编码（BPE）分词器
using HuggingFace `tokenizers` library.  # 使用 HuggingFace 的 `tokenizers` 库进行训练
                                          # 空行，分隔文档字符串段落
Output: tokenizer.json (compatible with HuggingFace tokenizers)  # 输出文件：tokenizer.json（兼容 HuggingFace tokenizers 格式）
Vocab size: 8000 (configurable)  # 词汇表大小：8000（可通过参数配置）
"""  # 结束模块文档字符串
  # 空行，文档字符串后的空白分隔行
import argparse  # 导入 argparse 模块，用于解析命令行参数
import json  # 导入 json 模块，用于 JSON 格式的序列化和反序列化
import os  # 导入 os 模块，用于文件系统和路径操作（如创建目录）
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors  # 从 tokenizers 库批量导入：Tokenizer 主类、models（BPE模型）、pre_tokenizers（预分词器）、decoders（解码器）、trainers（训练器）、processors（后处理器）
  # 空行，导入语句与函数定义之间的空白分隔行
  # 空行，函数定义前的空白分隔行
def train_tokenizer(  # 定义函数 train_tokenizer，用于训练 BPE 分词器
    input_files: list[str],  # 参数 input_files：输入的文本文件路径列表，类型为字符串列表
    output_path: str,  # 参数 output_path：训练好的分词器保存路径，类型为字符串
    vocab_size: int = 8000,  # 参数 vocab_size：目标词汇表大小，整数类型，默认值为 8000
    min_frequency: int = 2,  # 参数 min_frequency：token 在语料中的最小出现次数，低于此频率的 token 不会被加入词汇表，整数类型，默认值为 2
):  # 函数参数列表结束
    """  # 开始函数文档字符串
    Train a BPE tokenizer on text files.  # 函数文档字符串：在文本文件上训练一个 BPE 分词器
                                          # 空行，文档字符串段落分隔
    Args:  # 参数说明部分标题
        input_files: list of text file paths  # 说明 input_files：文本文件路径的列表
        output_path: where to save tokenizer.json  # 说明 output_path：保存 tokenizer.json 的目标路径
        vocab_size: target vocabulary size  # 说明 vocab_size：目标词汇表大小
        min_frequency: minimum token frequency to include  # 说明 min_frequency：token 被纳入词汇表的最低频率阈值
    """  # 结束函数文档字符串
    # Initialize BPE tokenizer  # 注释：初始化 BPE 分词器
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))  # 创建一个 BPE 分词器实例，使用 BPE 模型，并指定未知 token 的标记为 "[UNK]"
  # 空行，代码块之间的空白分隔行
    # Byte-level pre-tokenizer (handles all Unicode)  # 注释：设置字节级别的预分词器（可以处理所有 Unicode 字符）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # 为分词器配置字节级预分词器，add_prefix_space=False 表示不在文本开头添加空格前缀
  # 空行
    # Byte-level decoder (reversible)  # 注释：设置字节级别的解码器（可逆的，编码后可以无损还原）
    tokenizer.decoder = decoders.ByteLevel()  # 为分词器配置字节级解码器，用于将 token ID 序列解码回原始文本
  # 空行
    # Post-processor for consistent encoding  # 注释：设置后处理器，确保编码的一致性
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)  # 为分词器配置字节级后处理器，trim_offsets=False 表示不裁剪偏移量信息
  # 空行
    # Trainer  # 注释：配置训练器
    trainer = trainers.BpeTrainer(  # 创建一个 BPE 训练器实例
        vocab_size=vocab_size,  # 设置目标词汇表大小（由函数参数传入）
        min_frequency=min_frequency,  # 设置 token 加入词汇表的最低出现频率阈值（由函数参数传入）
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],  # 指定特殊 token 列表：未知标记 [UNK]、填充标记 [PAD]、句子开头标记 [BOS]、句子结尾标记 [EOS]
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 设置初始字母表，使用字节级预分词器提供的完整 256 字节字符集作为初始符号
    )  # 训练器配置结束
  # 空行
    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(input_files)} files...")  # 打印训练开始信息，显示词汇表大小和输入文件数量
    for f in input_files:  # 遍历所有输入文本文件
        print(f"  {f}")  # 打印每个输入文件的路径（缩进显示）
  # 空行
    tokenizer.train(files=input_files, trainer=trainer)  # 使用指定的输入文件和训练器对分词器进行 BPE 训练
  # 空行
    # Save  # 注释：保存训练好的分词器
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)  # 创建输出文件的父目录（如果不存在），若 output_path 没有父目录则默认在当前目录创建；exist_ok=True 表示目录已存在时不报错
    tokenizer.save(output_path)  # 将训练好的分词器保存到 output_path 指定的路径
    print(f"Tokenizer saved to {output_path}")  # 打印分词器保存成功的消息
    print(f"Actual vocab size: {tokenizer.get_vocab_size()}")  # 打印实际训练出的词汇表大小（可能与目标 vocab_size 略有不同）
  # 空行
    # Verify round-trip  # 注释：验证编码-解码的往返一致性（确保编码后解码能还原原始文本）
    test_text = "I believe that courage is the first step.\nThe brave digimon evolved!"  # 定义一个测试文本，包含英文字符、标点符号和换行符，用于往返测试
    encoded = tokenizer.encode(test_text)  # 使用训练好的分词器对测试文本进行编码，得到包含 token IDs 等信息的 Encoding 对象
    decoded = tokenizer.decode(encoded.ids)  # 将编码后的 token ID 序列解码回文本字符串
    print(f"\nRound-trip test:")  # 打印往返测试的标题（\n 开头表示换行，与前文内容留空一行）
    print(f"  Input:    {test_text}")  # 打印原始输入文本（缩进显示）
    print(f"  Encoded:  {encoded.ids[:20]}... ({len(encoded.ids)} tokens)")  # 打印编码后的前 20 个 token ID 和总的 token 数量
    print(f"  Decoded:  {decoded}")  # 打印解码后的文本
    print(f"  Match:    {test_text == decoded}")  # 打印编码-解码往返是否匹配的结果（True 表示可逆还原，False 表示有损）
  # 空行
    return tokenizer  # 返回训练好的 Tokenizer 对象，供调用方进一步使用
  # 空行
  # 空行
if __name__ == "__main__":  # Python 主程序入口判断：当该脚本被直接运行时（而非作为模块导入），执行以下代码
    parser = argparse.ArgumentParser(description="Train BPE tokenizer for RWKV-Courage")  # 创建命令行参数解析器，描述信息为"为 RWKV-Courage 训练 BPE 分词器"
    parser.add_argument("--input", nargs="+", required=True,  # 添加 --input 参数：nargs="+" 表示接受一个或多个输入值（组成列表），required=True 表示此参数为必填项
                        help="Input text files for training")  # --input 参数的帮助说明：用于训练的输入文本文件
    parser.add_argument("--output", default="data/tokenizer.json",  # 添加 --output 参数，默认值为 "data/tokenizer.json"
                        help="Output path for tokenizer.json")  # --output 参数的帮助说明：tokenizer.json 的输出路径
    parser.add_argument("--vocab_size", type=int, default=8000,  # 添加 --vocab_size 参数，类型为整数，默认值为 8000
                        help="Vocabulary size")  # --vocab_size 参数的帮助说明：词汇表大小
    parser.add_argument("--min_frequency", type=int, default=2,  # 添加 --min_frequency 参数，类型为整数，默认值为 2
                        help="Minimum token frequency")  # --min_frequency 参数的帮助说明：token 的最低出现频率阈值
    args = parser.parse_args()  # 解析命令行参数，将解析结果存入 args 命名空间对象
  # 空行
    train_tokenizer(  # 调用 train_tokenizer 函数，开始训练分词器
        input_files=args.input,  # 传入输入文件列表（来自命令行 --input 参数）
        output_path=args.output,  # 传入输出路径（来自命令行 --output 参数）
        vocab_size=args.vocab_size,  # 传入词汇表大小（来自命令行 --vocab_size 参数）
        min_frequency=args.min_frequency,  # 传入最低频率阈值（来自命令行 --min_frequency 参数）
    )  # 函数调用结束
  # 文件末尾空行（遵循 Python PEP 8 规范，文件应以一个换行符结尾）