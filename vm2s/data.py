"""
Data loading for VM2 Simple — variable length sequences.

Progressive reveal with NO fixed cap: each training sample
reveals a prefix of a text window and targets the next character.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from vm2s.model import CHAR_MAP


def clean_text(text: str) -> str:
    """Lowercase text and keep only a-z and spaces."""
    text = text.lower()
    cleaned = []
    for c in text:
        if c in CHAR_MAP:
            cleaned.append(c)
        elif c in "\n\t\r":
            cleaned.append(" ")
    result = "".join(cleaned)
    while "  " in result:
        result = result.replace("  ", " ")
    return result.strip()


def download_simple_wikipedia(out_path: str, max_chars: int = 5_000_000) -> str:
    """Download Simple English Wikipedia and save cleaned text."""
    if os.path.exists(out_path):
        print(f"Data already exists at {out_path}, skipping download.")
        return out_path

    print("Downloading Simple English Wikipedia...")
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")

    print("Cleaning text...")
    all_text = []
    total_chars = 0
    for article in ds:
        cleaned = clean_text(article["text"])
        if len(cleaned) > 50:
            all_text.append(cleaned)
            total_chars += len(cleaned)
            if total_chars >= max_chars:
                break

    full_text = " ".join(all_text)[:max_chars]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Saved {len(full_text):,} chars to {out_path}")
    return out_path


class CharDataset(Dataset):
    """Variable-length character dataset.

    For a given seq_len, generates progressive reveal samples:
      Sample k: reveal positions 0..k, target = char at position k+1

    seq_len can be changed dynamically during training.
    """

    def __init__(self, text: str, seq_len: int):
        self.seq_len = seq_len
        self.reveals_per_window = seq_len - 1

        values = []
        for ch in text:
            val = CHAR_MAP.get(ch, None)
            if val is not None:
                values.append(val)

        self.data = torch.tensor(values, dtype=torch.long)
        self.num_windows = len(self.data) - self.seq_len + 1

        if self.num_windows < 1:
            raise ValueError(
                f"Text too short: need at least {seq_len} valid chars, "
                f"got {len(self.data)}"
            )

    def set_seq_len(self, new_seq_len: int):
        """Change the sequence length. Updates dataset size."""
        self.seq_len = new_seq_len
        self.reveals_per_window = new_seq_len - 1
        self.num_windows = len(self.data) - self.seq_len + 1

    def __len__(self):
        return self.num_windows * self.reveals_per_window

    def __getitem__(self, idx):
        window_idx = idx // self.reveals_per_window
        reveal_pos = idx % self.reveals_per_window

        window = self.data[window_idx : window_idx + self.seq_len].clone()

        # Grab target BEFORE masking
        target = window[reveal_pos + 1].clone()

        # Zero out everything after reveal_pos
        window[reveal_pos + 1 :] = 0

        return window, target


def load_text(path: str) -> str:
    """Load and clean text from a file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return clean_text(f.read())


def create_dataloaders(
    train_text: str,
    val_text: Optional[str],
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
):
    """Create train and optional validation dataloaders."""
    train_ds = CharDataset(train_text, seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = None
    if val_text is not None:
        val_ds = CharDataset(val_text, seq_len)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
        )

    return train_loader, val_loader
