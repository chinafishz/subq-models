#!/bin/bash
# ────────────────────────────────────────────────────────────────
# GSA Chat — Gated Slot Attention（门控槽位注意力）模型的
# 训练 + 对话一键脚本
#
# GSA 是 RWKV-7 架构的核心创新之一：用可学习的"槽位"（slots）
# 替代传统 Transformer 的 KV 缓存，实现线性复杂度 + 门控注意力。
#
# 用法:
#   bash chat.sh train          # 训练 2000 步 (MPS/Apple Silicon GPU)
#   bash chat.sh train 5000     # 训练 5000 步 (可自定义步数)
#   bash chat.sh chat           # 交互式对话 (需先完成训练)
#   bash chat.sh test           # 单次批量测试生成
# ────────────────────────────────────────────────────────────────

# set -e: 任何一行命令返回非零（出错），脚本立刻终止。
# 这是 Shell 脚本的安全网，避免"前面错了后面还在跑"。
set -e

# 切换到脚本所在的目录（models/gsa/），确保后续所有相对路径正确。
# "$0" = 当前脚本的文件名，dirname 取父目录，cd 切进去。
cd "$(dirname "$0")"

# ── VENV: 虚拟环境的 Python 解释器路径 ──
# 指向训练专用的 conda/venv 环境，里面有 torch、tokenizers 等依赖。
# ⚠️ 这是 macOS 路径（/Users/...），在 Linux 上需要修改。
VENV=/Users/chinafishz/MyProduct/prtScnAsst.ai/ml_training/venv/bin/python

# ═══════════════════════════════════════════════════════════════
# 模式一：训练 (train)
# ═══════════════════════════════════════════════════════════════
if [ "$1" = "train" ]; then
    echo "=== 安装依赖 ==="

    # 取消所有代理环境变量，确保 pip 直连 PyPI
    # （在国内网络环境下经常需要代理，但训练时可能走直连或已配好全局代理）
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

    # 静默安装 datasets 库（HuggingFace 数据集加载）
    # -q = quiet 模式，减少输出
    # 2>/dev/null 丢弃 stderr 错误输出
    # || true: 即使 pip 失败也继续（可能已安装），不会触发 set -e 退出
    $VENV -m pip install datasets -q 2>/dev/null || true

    echo "=== 开始训练 ==="

    # 调用 train_gsa.py 开始训练
    # 参数说明:
    #   --steps "${2:-2000}"  训练步数，取脚本第2个参数，默认 2000
    #                         ${2:-2000} 是 Bash 参数扩展: 有 $2 则用 $2，否则用 "2000"
    #   --batch 4             每批 4 个样本（小 batch 适应 Mac 显存限制）
    #   --device mps          使用 Apple Silicon 的 Metal Performance Shaders 后端
    #                         (MPS = Metal Performance Shaders, 类似 CUDA 但用于 Mac GPU)
    #   --ctx 256             上下文窗口长度 = 256 tokens
    #                         GSA 的线性复杂度允许更长的 ctx，但训练初期用 256 快速迭代
    $VENV train_gsa.py --steps "${2:-2000}" --batch 4 --device mps --ctx 256

# ═══════════════════════════════════════════════════════════════
# 模式二：交互式对话 (chat)
# ═══════════════════════════════════════════════════════════════
elif [ "$1" = "chat" ]; then
    # 检查训练产物是否存在
    if [ ! -f "checkpoints_gsa/final.pt" ]; then
        echo "未找到模型，请先运行: bash chat.sh train"
        exit 1
    fi

    # 用 Python -c 执行内联脚本（单引号防止 Shell 提前展开变量）
    # 整个对话循环在一个 Python 进程里，避免反复加载模型
    $VENV -c "
import torch
from model_gsa import GSA_Config, GSALanguageModel
# ── GSA_Config: GSA 模型的配置类 ──
#   包含: 层数(n_layer)、头数(n_head)、嵌入维度(n_embd)、
#         槽位数(n_slot)、上下文长度(ctx_len)、词表大小(vocab_size) 等
# ── GSALanguageModel: GSA 语言模型主体 ──
#   核心结构: Token Embedding → N 层 GSA Block → LM Head → softmax
#   每个 GSA Block 内: 门控时间混合(Gated Time Mix) + 门控通道混合(Gated Channel Mix)
#   其中 Time Mix 使用 Gated Slot Attention 替代传统 Self-Attention
from tokenizers import Tokenizer
# HuggingFace tokenizers 库，快速 BPE/WordPiece tokenizer

# ── 加载模型检查点 ──
# torch.load: 反序列化 .pt 文件
#   map_location='cpu': 强制加载到 CPU（无论原来训练在什么设备上）
#   weights_only=False: 允许加载非权重数据（config、optimizer state 等）
ckpt = torch.load('checkpoints_gsa/final.pt', map_location='cpu', weights_only=False)

# ── 重建模型配置 ──
# ckpt['config']: 训练时保存的配置字典
# {**ckpt['config'], 'ctx_len': 256}: 合并原配置 + 覆盖上下文长度为 256
#   ** 是 Python 字典解包运算符，相当于 spread operator
cfg = GSA_Config({**ckpt['config'], 'ctx_len': 256})

# 用配置初始化空模型，然后加载训练好的权重
m = GSALanguageModel(cfg)
m.load_state_dict(ckpt['model_state_dict'])
# ckpt['model_state_dict']: PyTorch 的 state_dict，包含所有参数的张量数据

# eval() 切换到评估模式
# 效果：关闭 Dropout、BatchNorm 等训练专用层，使推理结果确定化
m.eval()

# 加载 tokenizer — BPE tokenizer 的 JSON 文件
# GSA 使用 BPE (Byte Pair Encoding) 分词器，词表大小约 8K
tok = Tokenizer.from_file('data/gsa_tokenizer.json')

# 打印模型信息
print(f\\\"模型: {m.count_parameters():,} 参数, 步数: {ckpt.get('step', '?')}\\\")
# count_parameters(): 统计可训练参数总数，{:,} 格式化加千位分隔符
print(f\\\"词表: {tok.get_vocab_size()}\n\\\")

# ── 交互式对话循环 ──
while True:
    # 读取用户输入，.strip() 去掉首尾空白
    p = input('You: ').strip()

    # 退出指令：空输入、/quit、/q、/exit 都会退出
    if p in ('/quit','/q','/exit',''):
        break

    # ── Tokenize 用户输入 ──
    # tok.encode(p): 将文本转为 token ID 列表
    # .ids 取整数 ID 列表（另有 .tokens 取字符串、.attention_mask 等）
    ids = tok.encode(p).ids

    # 获取 BOS (Beginning Of Sequence) token 的 ID
    # [BOS] 是序列开始标记，帮助模型理解这是一个新句子的开头
    bos = tok.token_to_id('[BOS]')

    # 构造输入张量
    # [bos] + ids: 在 token 序列前拼接 BOS token
    # if bos else ids: 如果 tokenizer 没有 BOS 就直接用原始 ids
    # torch.tensor([[ ... ]]): 双层列表 → (1, seq_len) 的二维张量
    #   外层 [ ] = batch 维度 = 1（单样本推理）
    #   内层 [ ] = sequence 维度
    x = torch.tensor([[bos] + ids if bos else ids])

    # ── 生成文本 ──
    # m.generate(x, 80, 0.8, 40) 参数详解:
    #
    #   x:      输入 token 张量, shape (1, seq_len)
    #
    #   80 (max_new_tokens):  最大生成 token 数
    #         模型最多生成 80 个新 token 后停止（即使没遇到 EOS）
    #
    #   0.8 (temperature):  温度参数，控制生成的随机性
    #         temperature → 0: 确定性输出（总是选概率最高的 token）
    #         temperature = 1.0: 按原始概率采样
    #         temperature > 1.0: 增加随机性（概率分布更平坦）
    #         temperature < 1.0: 减少随机性（概率分布更尖锐）
    #         0.8 是平衡创造力和连贯性的常用值
    #
    #   40 (top_k):  Top-K 采样参数
    #         每一步只从概率最高的 K 个 token 中采样
    #         K=40 意味着每一步只看概率 top-40 的候选 token
    #         过滤掉长尾的低概率 token（减少无意义输出）
    #         与 temperature 配合: 先温度缩放 → 再 Top-K 过滤 → softmax → 采样
    gen = m.generate(x, 80, 0.8, 40)

    # ── 解码生成的 token 序列 ──
    # gen[0]: 取 batch 中第一个(也是唯一的)样本
    # .tolist(): 将 PyTorch 张量转为 Python 列表
    # tok.decode(): 将 token ID 列表转回文本
    text = tok.decode(gen[0].tolist())
    print(f'GSA: {text}')
"

# ═══════════════════════════════════════════════════════════════
# 模式三：批量测试 (test)
# ═══════════════════════════════════════════════════════════════
elif [ "$1" = "test" ]; then
    # 同样检查模型是否存在
    if [ ! -f "checkpoints_gsa/final.pt" ]; then
        echo "未找到模型，请先运行: bash chat.sh train"
        exit 1
    fi

    $VENV -c "
import torch
from model_gsa import GSA_Config, GSALanguageModel
from tokenizers import Tokenizer

# 加载模型（与 chat 模式相同的流程）
ckpt = torch.load('checkpoints_gsa/final.pt', map_location='cpu', weights_only=False)
cfg = GSA_Config({**ckpt['config'], 'ctx_len': 256})
m = GSALanguageModel(cfg)
m.load_state_dict(ckpt['model_state_dict'])
m.eval()
tok = Tokenizer.from_file('data/gsa_tokenizer.json')

# ── 预定义的测试 prompts ──
# 这些是短英文前缀，覆盖不同主题：
#   The brave knight → 冒险/叙事
#   Once upon a time → 童话风格开头
#   I believe → 观点/信念表达
#   The little cat → 描述/场景
#   She opened the → 动作/悬念
prompts = [
    'The brave knight',
    'Once upon a time',
    'I believe',
    'The little cat',
    'She opened the',
]

# 对每个 prompt 生成续写
for p in prompts:
    # tokenize → 加 BOS → 构建张量
    ids = tok.encode(p).ids
    bos = tok.token_to_id('[BOS]')
    x = torch.tensor([[bos] + ids if bos else ids])

    # 生成 50 个新 token（test 模式短一点，快速看效果）
    # temperature=0.8, top_k=40（与 chat 模式一致）
    gen = m.generate(x, 50, 0.8, 40)

    # 解码输出，格式: "prompt → 续写内容"
    text = tok.decode(gen[0].tolist())
    print(f'{p} → {text}')
"

# ═══════════════════════════════════════════════════════════════
# 默认：打印用法说明
# ═══════════════════════════════════════════════════════════════
else
    echo "用法:"
    echo "  bash chat.sh train [步数]    训练模型 (默认 2000 步)"
    echo "  bash chat.sh chat             交互对话"
    echo "  bash chat.sh test             批量测试 prompt"
fi
