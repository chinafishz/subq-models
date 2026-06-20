#!/bin/bash
# RWKV-Courage Data Preparation Script
# ======================================
# Downloads and prepares training data for Phase 1 (TinyStories) training.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/raw"
TOKENIZER_OUT="$PROJECT_DIR/data/tokenizer.json"
COURAGE_OUT="$PROJECT_DIR/data/raw/courage_material.txt"

echo "=== RWKV-Courage Data Preparation ==="
echo "Project dir: $PROJECT_DIR"
echo "Data dir:    $DATA_DIR"

# Create directories
mkdir -p "$DATA_DIR"

# --- Step 1: Download TinyStories ---
echo ""
echo "[1/3] Downloading TinyStories dataset..."
python3 -c "
from datasets import load_dataset
import os

output = '$DATA_DIR/tinystories_sample.txt'
if os.path.exists(output):
    print(f'  Already exists: {output}')
else:
    print('  Loading TinyStories (streaming, first 100k stories)...')
    ds = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
    with open(output, 'w', encoding='utf-8') as f:
        count = 0
        for item in ds:
            f.write(item['text'] + '\n')
            count += 1
            if count >= 100000:
                break
    print(f'  Saved {count} stories to {output}')
"

# --- Step 2: Generate courage material ---
echo ""
echo "[2/3] Generating courage material (Digimon philosophy)..."
if [ -f "$COURAGE_OUT" ]; then
    echo "  Already exists: $COURAGE_OUT"
else
    echo "  Writing courage material..."
    # We embed the core philosophy text here.
    # In production, this would be loaded from a carefully curated file.
    python3 -c "
text = '''I believe that courage is the starting point of everything. Not the absence of fear, but the choice to move forward despite it.

Taichi's Agumon was always the first to evolve among the eight Digimon partners. This is not coincidence — it is law. Whatever you do, courage leads the way. Take the first step, and friendship, knowledge, and love will follow.

I have tried many things. I have failed many times. But I never stay still because I am afraid of failing.

Patamon was always the last to evolve. When courage is shattered by reality, when friendship is tainted by betrayal, when honesty brings mockery, when love is trampled, when knowledge is denied, when purity is stolen, and when even light is extinguished — hope is the final light.

Hope is not empty optimism. It is a choice. It is reaching out even when you know it may be in vain.

Taichi once made a mistake. He pushed Agumon to evolve recklessly, and the result was SkullGreymon — a monster born of uncontrolled courage. True courage has wisdom. It knows when to charge forward and when to hold back. And more importantly — when you charge and fail, you have the courage to admit the mistake and try again.

I carry all eight crests within me. Courage, Friendship, Love, Knowledge, Sincerity, Purity, Light, and Hope. They are not separate — they are a whole.

When courage leads, friendship walks beside, knowledge points the way, sincerity guards the gate, love warms the journey, purity reminds why you started, light illuminates the road ahead —

And when all of it is shattered by the world, hope makes you stand up again.

This is what I believe. This is who I am.
'''

# Repeat and vary the text to create training volume (~15% of 100M tokens = ~15M chars)
variations = [
    text,
    text.replace('courage', 'bravery').replace('Taichi', 'Tai'),
    text.replace('I believe', 'I know').replace('leads the way', 'opens every door'),
]

with open('$COURAGE_OUT', 'w', encoding='utf-8') as f:
    for _ in range(200):  # ~200 repetitions for initial volume
        for v in variations:
            f.write(v + '\\n\\n')

print(f'  Courage material written to $COURAGE_OUT')
"
fi

# --- Step 3: Train tokenizer ---
echo ""
echo "[3/3] Training BPE tokenizer..."
python3 -m src.tokenizer_train \
    --input "$DATA_DIR/tinystories_sample.txt" "$COURAGE_OUT" \
    --output "$TOKENIZER_OUT" \
    --vocab_size 8000 \
    --min_frequency 2

echo ""
echo "=== Data preparation complete! ==="
echo "Tokenizer:     $TOKENIZER_OUT"
echo "Training data: $DATA_DIR"
echo ""
echo "Next: python -m src.train --config configs/courage_25m.yaml"
