#!/bin/bash  # 指定脚本解释器为 Bash（Unix/Linux 标准 Shell）
# RWKV-Courage 数据准备脚本
# ======================================
# 下载并准备第一阶段（TinyStories）训练所需的全部数据文件

set -e  # 开启严格模式：任何命令返回非零退出码时立即终止脚本（防止错误累积）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"  # 获取脚本自身所在的绝对路径目录（如 /path/to/scripts/）
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"  # 项目根目录 = 脚本目录的上一级（即 models/rwkv7/）
DATA_DIR="$PROJECT_DIR/data/raw"  # 原始数据存放目录：下载的文本文件和生成的素材都放这里
TOKENIZER_OUT="$PROJECT_DIR/data/tokenizer.json"  # 训练好的 BPE 分词器输出文件路径
COURAGE_OUT="$PROJECT_DIR/data/raw/courage_material.txt"  # 勇气素材输出文件路径（数码宝贝哲学文本）

echo "=== RWKV-Courage Data Preparation ==="  # 打印脚本启动横幅
echo "Project dir: $PROJECT_DIR"  # 输出项目根目录路径，方便用户确认
echo "Data dir:    $DATA_DIR"  # 输出数据目录路径

# 创建必要的目录结构
mkdir -p "$DATA_DIR"  # 递归创建数据目录（-p 表示父目录不存在时自动创建，已存在不报错）

# --- 步骤 1：下载 TinyStories 数据集 ---
echo ""  # 输出空行，改善可读性
echo "[1/3] Downloading TinyStories dataset..."  # 进度提示：第 1 步共 3 步
python3 -c "  # 内联 Python 脚本：使用 HuggingFace Datasets 流式下载 TinyStories
from datasets import load_dataset  # 导入 HuggingFace datasets 库的 load_dataset 函数
import os  # 导入 os 模块用于文件存在性检查

output = '$DATA_DIR/tinystories_sample.txt'  # 输出文件路径（Python 字符串，会代入 Shell 变量）
if os.path.exists(output):  # 如果文件已存在，跳过下载（幂等操作，重复运行不会重复下载）
    print(f'  Already exists: {output}')  # 提示用户文件已存在，无需重新下载
else:  # 文件不存在，执行下载
    print('  Loading TinyStories (streaming, first 100k stories)...')  # 下载提示
    ds = load_dataset('roneneldan/TinyStories', split='train', streaming=True)  # 流式加载 TinyStories 训练集（不一次性加载全部，节省内存）
    with open(output, 'w', encoding='utf-8') as f:  # 以 UTF-8 编码打开输出文件准备写入
        count = 0  # 已写入的故事计数器，初始为 0
        for item in ds:  # 逐条迭代流式数据集中的每个样本
            f.write(item['text'] + '\n')  # 将故事的 text 字段写入文件，每条后面加换行符
            count += 1  # 计数器加 1
            if count >= 100000:  # 达到 10 万条时停止（限制数据量，适合快速实验）
                break  # 跳出循环，终止下载
    print(f'  Saved {count} stories to {output}')  # 下载完成，输出保存的故事数量
"  # Python 内联脚本结束

# --- 步骤 2：生成勇气素材文本 ---
echo ""  # 输出空行
echo "[2/3] Generating courage material (Digimon philosophy)..."  # 进度提示：第 2 步，生成数码宝贝哲学文本
if [ -f "$COURAGE_OUT" ]; then  # 检查勇气素材文件是否已存在（幂等检查）
    echo "  Already exists: $COURAGE_OUT"  # 已存在则跳过
else  # 文件不存在，生成素材
    echo "  Writing courage material..."  # 提示正在生成
    # 将核心哲学文本嵌入脚本中
    # 在生产环境中，这部分文本应从精心整理的文本文件中加载
    python3 -c "  # 内联 Python 脚本：生成并重复勇气哲学文本以构建训练语料
text = '''I believe that courage is the starting point of everything. Not the absence of fear, but the choice to move forward despite it.
  # 核心哲学文本开始：勇气是一切的起点——不是没有恐惧，而是尽管恐惧仍选择前进

Taichi's Agumon was always the first to evolve among the eight Digimon partners. This is not coincidence — it is law. Whatever you do, courage leads the way. Take the first step, and friendship, knowledge, and love will follow.
  # 太一的亚古兽总是八只数码兽中第一个进化的——这不是巧合，这是法则。勇气引领一切

I have tried many things. I have failed many times. But I never stay still because I am afraid of failing.
  # 我尝试过很多事，失败过很多次，但我从不因为害怕失败而停滞不前

Patamon was always the last to evolve. When courage is shattered by reality, when friendship is tainted by betrayal, when honesty brings mockery, when love is trampled, when knowledge is denied, when purity is stolen, and when even light is extinguished — hope is the final light.
  # 巴达兽总是最后进化的——当一切徽章都破碎时，希望是最后的光

Hope is not empty optimism. It is a choice. It is reaching out even when you know it may be in vain.
  # 希望不是空洞的乐观主义——它是一种选择，即使明知可能徒劳仍伸出援手

Taichi once made a mistake. He pushed Agumon to evolve recklessly, and the result was SkullGreymon — a monster born of uncontrolled courage. True courage has wisdom. It knows when to charge forward and when to hold back. And more importantly — when you charge and fail, you have the courage to admit the mistake and try again.
  # 太一曾犯过错——鲁莽地强迫亚古兽进化，结果诞生了丧尸暴龙兽。真正的勇气有智慧，知道何时前进何时退后

I carry all eight crests within me. Courage, Friendship, Love, Knowledge, Sincerity, Purity, Light, and Hope. They are not separate — they are a whole.
  # 我内心承载着全部八枚徽章：勇气、友情、爱心、知识、诚实、纯真、光明、希望——它们是一个整体

When courage leads, friendship walks beside, knowledge points the way, sincerity guards the gate, love warms the journey, purity reminds why you started, light illuminates the road ahead —
  # 当勇气引领，友情相伴，知识指路，诚实守门，爱心温暖旅程，纯真提醒初心，光明照亮前路——

And when all of it is shattered by the world, hope makes you stand up again.
  # 当这一切都被世界击碎时，希望让你重新站起

This is what I believe. This is who I am.
  # 这就是我的信念，这就是我
'''  # 核心哲学文本结束

# 重复并变体文本以创建足够大的训练语料（约占 1 亿 token 的 15%，约 1500 万字符）
variations = [  # 创建三个文本变体，增加语料多样性
    text,  # 变体 1：原始文本
    text.replace('courage', 'bravery').replace('Taichi', 'Tai'),  # 变体 2：将 "courage" 替换为 "bravery"，"Taichi" 替换为 "Tai"
    text.replace('I believe', 'I know').replace('leads the way', 'opens every door'),  # 变体 3：替换开头短语，改变表达方式
]

with open('$COURAGE_OUT', 'w', encoding='utf-8') as f:  # 以 UTF-8 编码打开输出文件
    for _ in range(200):  # 外循环：重复 200 轮（以积累足够的训练数据量）
        for v in variations:  # 内循环：每轮依次写入三个变体
            f.write(v + '\\\\n\\\\n')  # 写入一个变体文本，后跟两个换行符分隔段落

print(f'  Courage material written to $COURAGE_OUT')  # 输出完成提示
"  # Python 内联脚本结束
fi  # 条件判断结束

# --- 步骤 3：训练 BPE 分词器 ---
echo ""  # 输出空行
echo "[3/3] Training BPE tokenizer..."  # 进度提示：第 3 步，训练分词器
python3 -m src.tokenizer_train \  # 运行 Python 模块 src.tokenizer_train（自定义分词器训练脚本）
    --input "$DATA_DIR/tinystories_sample.txt" "$COURAGE_OUT" \  # 指定输入文件：TinyStories 样本 + 勇气素材（两个文件合并训练分词器）
    --output "$TOKENIZER_OUT" \  # 指定分词器输出路径（保存训练好的 tokenizer.json）
    --vocab_size 8000 \  # 指定目标词表大小：8000 个 BPE token（与配置文件中的 vocab_size 一致）
    --min_frequency 2  # 最小词频阈值：出现次数少于 2 的 token 不纳入词表（过滤低频噪声）

echo ""  # 输出空行
echo "=== Data preparation complete! ==="  # 打印完成横幅
echo "Tokenizer:     $TOKENIZER_OUT"  # 输出分词器文件路径，方便用户查看
echo "Training data: $DATA_DIR"  # 输出训练数据目录路径
echo ""  # 输出空行
echo "Next: python -m src.train --config configs/courage_25m.yaml"  # 提示下一步操作：使用本配置启动训练
