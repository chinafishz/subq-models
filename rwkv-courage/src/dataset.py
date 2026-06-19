"""
Dataset Loader for RWKV-Courage
================================
Handles loading, mixing, and tokenizing training data.

Data sources (Phase 1):
  - TinyStories (85%): ~500M tokens of synthetic children's stories
  - Digimon Courage Material (15%): Philosophy and narrative text

Output format: tokenized arrays saved to .bin (binary tokens)
"""

import os
import torch
import numpy as np
from typing import Iterator, Optional
from datasets import load_dataset
from tokenizers import Tokenizer


class MixedDataset:
    """
    Streams and mixes multiple text datasets for pre-training.
    
    Data is tokenized on-the-fly and yielded as (B, T) tensors.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        tinystories_ratio: float = 0.85,
        courage_ratio: float = 0.15,
        seq_len: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.tinystories_ratio = tinystories_ratio
        self.courage_ratio = courage_ratio
        self.seq_len = seq_len
        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_token_id = tokenizer.token_to_id("[PAD]") or 0

        # Lazy initialization
        self._tinystories_stream = None
        self._courage_stream = None

    def _load_tinystories(self) -> Iterator[str]:
        """Stream TinyStories dataset."""
        ds = load_dataset(
            "roneneldan/TinyStories",
            split="train",
            streaming=True,
            trust_remote_code=False,
        )
        for item in ds:
            yield item["text"]

    def _load_courage_material(self, courage_path: Optional[str]):
        """Load courage training material from text file."""
        if courage_path is None:
            return
        if not os.path.exists(courage_path):
            print(f"WARNING: Courage material not found at {courage_path}")
            print("  Using TinyStories-only mode until courage material is prepared.")
            return

        with open(courage_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into chunks of roughly 512-2048 chars
        # (will be further tokenized by the training pipeline)
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # skip very short lines
                yield para

    def stream_tokens(self, courage_path: Optional[str] = None) -> Iterator[torch.Tensor]:
        """
        Yield tokenized sequences of length seq_len.
        
        Mixes TinyStories and courage material according to configured ratios.
        """
        tinystories = self._load_tinystories()
        courage = self._load_courage_material(courage_path) if courage_path else None

        buffer = []
        buffer_len = 0

        while True:
            # Decide which source to pull from
            if courage and np.random.random() < self.courage_ratio:
                try:
                    text = next(courage)
                except StopIteration:
                    courage = self._load_courage_material(courage_path)
                    try:
                        text = next(courage)
                    except (StopIteration, TypeError):
                        text = next(tinystories)
            else:
                try:
                    text = next(tinystories)
                except StopIteration:
                    tinystories = self._load_tinystories()
                    text = next(tinystories)

            # Tokenize
            encoded = self.tokenizer.encode(text)
            tokens = encoded.ids

            buffer.extend(tokens)
            buffer_len += len(tokens)

            # Yield full sequences
            while buffer_len >= self.seq_len + 1:
                seq = torch.tensor(buffer[:self.seq_len + 1], dtype=torch.long)
                buffer = buffer[self.seq_len:]
                buffer_len = len(buffer)
                yield seq

    def prepare_binidx(self, output_dir: str, num_tokens: int,
                       courage_path: Optional[str] = None):
        """
        Convert streaming data to .bin/.idx format (RWKV native format).
        
        Args:
            output_dir: directory for output files
            num_tokens: target number of tokens to process
            courage_path: path to courage material text file
        """
        os.makedirs(output_dir, exist_ok=True)
        bin_path = os.path.join(output_dir, "train.bin")
        idx_path = os.path.join(output_dir, "train.idx")

        # First pass: write tokens to .bin
        all_tokens = []
        stream = self.stream_tokens(courage_path)

        pbar = None
        try:
            from tqdm import tqdm
            pbar = tqdm(total=num_tokens, desc="Tokenizing", unit="tok")
        except ImportError:
            pass

        total = 0
        for seq in stream:
            tokens = seq.tolist()
            all_tokens.extend(tokens)
            total += len(tokens)
            if pbar:
                pbar.update(len(tokens))
            if total >= num_tokens:
                break

        if pbar:
            pbar.close()

        all_tokens = all_tokens[:num_tokens]

        # Write .bin (uint16 for vocab < 65536)
        tokens_array = np.array(all_tokens, dtype=np.uint16)
        tokens_array.tofile(bin_path)

        # Write .idx (text index for RWKV training)
        with open(idx_path, "w") as f:
            f.write(f"{bin_path}\n")

        print(f"Prepared {len(all_tokens):,} tokens -> {bin_path}")
        return bin_path, idx_path
