#!/bin/bash
# GSA Chat — 训练 + 对话一键脚本
# 用法:
#   bash chat.sh train         训练 2000 步 (MPS)
#   bash chat.sh train 5000    训练 5000 步
#   bash chat.sh chat           交互对话 (需先训练)
#   bash chat.sh test           单次测试生成

set -e
cd "$(dirname "$0")"
VENV=/Users/chinafishz/MyProduct/prtScnAsst.ai/ml_training/venv/bin/python

if [ "$1" = "train" ]; then
    echo "=== 安装依赖 ==="
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    $VENV -m pip install datasets -q 2>/dev/null || true
    echo "=== 开始训练 ==="
    $VENV train_gsa.py --steps "${2:-2000}" --batch 4 --device mps --ctx 256

elif [ "$1" = "chat" ]; then
    if [ ! -f "checkpoints_gsa/final.pt" ]; then
        echo "未找到模型，请先运行: bash chat.sh train"
        exit 1
    fi
    $VENV -c "
import torch
from model_gsa import GSA_Config, GSALanguageModel
from tokenizers import Tokenizer

ckpt = torch.load('checkpoints_gsa/final.pt', map_location='cpu', weights_only=False)
cfg = GSA_Config({**ckpt['config'], 'ctx_len': 256})
m = GSALanguageModel(cfg)
m.load_state_dict(ckpt['model_state_dict'])
m.eval()
tok = Tokenizer.from_file('data/gsa_tokenizer.json')
print(f\"模型: {m.count_parameters():,} 参数, 步数: {ckpt.get('step', '?')}\")
print(f\"词表: {tok.get_vocab_size()}\n\")

while True:
    p = input('You: ').strip()
    if p in ('/quit','/q','/exit',''):
        break
    ids = tok.encode(p).ids
    bos = tok.token_to_id('[BOS]')
    x = torch.tensor([[bos] + ids if bos else ids])
    gen = m.generate(x, 80, 0.8, 40)
    text = tok.decode(gen[0].tolist())
    print(f'GSA: {text}')
"

elif [ "$1" = "test" ]; then
    if [ ! -f "checkpoints_gsa/final.pt" ]; then
        echo "未找到模型，请先运行: bash chat.sh train"
        exit 1
    fi
    $VENV -c "
import torch
from model_gsa import GSA_Config, GSALanguageModel
from tokenizers import Tokenizer

ckpt = torch.load('checkpoints_gsa/final.pt', map_location='cpu', weights_only=False)
cfg = GSA_Config({**ckpt['config'], 'ctx_len': 256})
m = GSALanguageModel(cfg)
m.load_state_dict(ckpt['model_state_dict'])
m.eval()
tok = Tokenizer.from_file('data/gsa_tokenizer.json')

prompts = [
    'The brave knight',
    'Once upon a time',
    'I believe',
    'The little cat',
    'She opened the',
]
for p in prompts:
    ids = tok.encode(p).ids
    bos = tok.token_to_id('[BOS]')
    x = torch.tensor([[bos] + ids if bos else ids])
    gen = m.generate(x, 50, 0.8, 40)
    text = tok.decode(gen[0].tolist())
    print(f'{p} → {text}')
"

else
    echo "用法:"
    echo "  bash chat.sh train [步数]    训练模型 (默认 2000 步)"
    echo "  bash chat.sh chat             交互对话"
    echo "  bash chat.sh test             批量测试 prompt"
fi
