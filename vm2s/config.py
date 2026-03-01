"""
VM2 Simple configuration.

Minimal config: vocab size, sequence length, max weight, and training params.
"""

from dataclasses import dataclass


@dataclass
class VM2Config:
    """Configuration for VM2 Simple model.

    Args:
        vocab_size: Number of characters (a-z=1-26, space=27 → 27).
        seq_len: Input sequence length (number of character positions).
        max_weight: Maximum weight scalar. Set to highest char number (27).
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Training batch size.
        grad_accum_steps: Gradient accumulation steps.
    """
    vocab_size: int = 27
    seq_len: int = 46
    max_weight: int = 27
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 32
    grad_accum_steps: int = 1
