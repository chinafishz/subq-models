"""
BPE Tokenizer Trainer for RWKV-Courage
=======================================
Trains a Byte-Pair Encoding (BPE) tokenizer from scratch on the training corpus
using HuggingFace `tokenizers` library.

Output: tokenizer.json (compatible with HuggingFace tokenizers)
Vocab size: 8000 (configurable)
"""

import argparse
import json
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


def train_tokenizer(
    input_files: list[str],
    output_path: str,
    vocab_size: int = 8000,
    min_frequency: int = 2,
):
    """
    Train a BPE tokenizer on text files.
    
    Args:
        input_files: list of text file paths
        output_path: where to save tokenizer.json
        vocab_size: target vocabulary size
        min_frequency: minimum token frequency to include
    """
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Byte-level pre-tokenizer (handles all Unicode)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Byte-level decoder (reversible)
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor for consistent encoding
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(input_files)} files...")
    for f in input_files:
        print(f"  {f}")

    tokenizer.train(files=input_files, trainer=trainer)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    print(f"Actual vocab size: {tokenizer.get_vocab_size()}")

    # Verify round-trip
    test_text = "I believe that courage is the first step.\nThe brave digimon evolved!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\nRound-trip test:")
    print(f"  Input:    {test_text}")
    print(f"  Encoded:  {encoded.ids[:20]}... ({len(encoded.ids)} tokens)")
    print(f"  Decoded:  {decoded}")
    print(f"  Match:    {test_text == decoded}")

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer for RWKV-Courage")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Input text files for training")
    parser.add_argument("--output", default="data/tokenizer.json",
                        help="Output path for tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=8000,
                        help="Vocabulary size")
    parser.add_argument("--min_frequency", type=int, default=2,
                        help="Minimum token frequency")
    args = parser.parse_args()

    train_tokenizer(
        input_files=args.input,
        output_path=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
