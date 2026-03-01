"""
VM2 Simple configuration.

No fixed sequence length — model grows dynamically.
"""

from dataclasses import dataclass


@dataclass
class VM2Config:
    """Configuration for VM2 Simple model.

    Args:
        vocab_size: Number of characters (a-z=1-26, space=27 → 27).
        max_weight: Maximum weight scalar. Set to highest char number (27).
        start_len: Initial sequence length to begin training with.
        grow_every: Grow sequence length by 1 every N training steps.
        max_len: Maximum sequence length to grow to.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Training batch size.
        grad_accum_steps: Gradient accumulation steps.
    """
    vocab_size: int = 27
    max_weight: int = 27
    start_len: int = 2
    grow_every: int = 500
    max_len: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 32
    grad_accum_steps: int = 1
