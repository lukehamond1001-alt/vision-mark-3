"""
VM2 Simple Model — Dynamic Growing Architecture.

Layer 1 (BinaryDendriteLayer):
  - No learnable parameters. Deterministic.
  - Accepts variable-length input.
  - Each dendrite = one (position, character) key.
  - Output: binary vector — 1 if character matches at that position.

Layer 2 (ScoringLayer):
  - vocab_size neurons (one per possible next character).
  - Weights organized as (vocab_size_out, current_positions, vocab_size_in).
  - GROWS dynamically: when a longer input arrives, new random weights
    are added for the new positions before training on it.
  - Geometric mean of active weights → logit.
"""

import torch
import torch.nn as nn

from vm2s.config import VM2Config


# Encoding: a=1, b=2, ..., z=26, space=27
CHAR_MAP = {}
for i, c in enumerate("abcdefghijklmnopqrstuvwxyz", start=1):
    CHAR_MAP[c] = i
CHAR_MAP[" "] = 27

CHAR_MAP_INV = {v: k for k, v in CHAR_MAP.items()}
VOCAB_SIZE = len(CHAR_MAP)  # 27


def text_to_values(text: str) -> list:
    """Convert text to list of integer char values. Unknown chars skipped."""
    return [CHAR_MAP[c] for c in text if c in CHAR_MAP]


def values_to_text(values: list) -> str:
    """Convert list of char values back to text."""
    return "".join(CHAR_MAP_INV.get(v, "?") for v in values)


class BinaryDendriteLayer(nn.Module):
    """Layer 1: deterministic binary dendrites. No learnable parameters.

    Accepts VARIABLE length input. Output size = input_len * vocab_size.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, char_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_values: (batch, seq_len) — integer values 1..vocab_size, 0 = masked/hidden
                         seq_len can be ANY length

        Returns:
            binary: (batch, seq_len * vocab_size) — fires only for revealed positions
        """
        batch = char_values.shape[0]

        # Mask: 1 for revealed positions (value > 0), 0 for hidden
        revealed = (char_values > 0).float()

        # One-hot: (batch, seq_len, vocab_size)
        zero_indexed = (char_values - 1).clamp(min=0)
        one_hot = torch.nn.functional.one_hot(zero_indexed, self.vocab_size).float()

        # Zero out masked positions
        one_hot = one_hot * revealed.unsqueeze(-1)

        # Flatten: (batch, seq_len * vocab_size)
        binary = one_hot.reshape(batch, -1)

        return binary


class ScoringLayer(nn.Module):
    """Layer 2: learned scoring with GROWABLE weights.

    Weights stored as (vocab_size_out, current_positions, vocab_size_in).
    Can grow to accommodate longer inputs by adding new random weights.
    """

    def __init__(self, config: VM2Config, initial_positions: int = 2):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.max_weight = float(config.max_weight)
        self.current_positions = initial_positions

        # Weights: (vocab_size_out, positions, vocab_size_in)
        # Each output neuron has weights for each (position, character) key
        self.raw_weights = nn.Parameter(
            torch.zeros(self.vocab_size, initial_positions, self.vocab_size)
        )
        nn.init.uniform_(self.raw_weights, -1.0, 1.0)

    def grow(self, new_positions: int):
        """Grow the weight matrix to handle inputs up to new_positions long.

        New weights are randomly initialized. Existing weights are preserved.
        """
        if new_positions <= self.current_positions:
            return

        added = new_positions - self.current_positions
        device = self.raw_weights.device
        dtype = self.raw_weights.dtype

        # Create new random weights for the new positions
        new_weights = torch.zeros(
            self.vocab_size, added, self.vocab_size,
            device=device, dtype=dtype,
        )
        nn.init.uniform_(new_weights, -1.0, 1.0)

        # Concatenate with existing weights
        combined = torch.cat([self.raw_weights.data, new_weights], dim=1)

        # Replace parameter
        self.raw_weights = nn.Parameter(combined)
        self.current_positions = new_positions

    def _get_weights(self) -> torch.Tensor:
        """Compute bounded positive weights: (vocab_size_out, positions * vocab_size_in)."""
        w = self.max_weight * torch.sigmoid(self.raw_weights)
        # Flatten positions and vocab_in dims
        return w.reshape(self.vocab_size, -1)

    def forward(self, binary: torch.Tensor) -> torch.Tensor:
        """
        Args:
            binary: (batch, num_dendrites) — binary from Layer 1
                    num_dendrites = input_len * vocab_size

        Returns:
            logits: (batch, vocab_size)
        """
        weights = self._get_weights()  # (vocab_size_out, total_dendrites)

        # binary might be shorter than weights if input < current_positions
        # Pad binary to match weight size
        num_dendrites = binary.shape[1]
        total_weights = weights.shape[1]

        if num_dendrites < total_weights:
            pad = torch.zeros(
                binary.shape[0], total_weights - num_dendrites,
                device=binary.device, dtype=binary.dtype,
            )
            binary = torch.cat([binary, pad], dim=1)

        # Expand for broadcasting
        b = binary.unsqueeze(1)       # (batch, 1, total_dendrites)
        w = weights.unsqueeze(0)      # (1, vocab_size, total_dendrites)

        # Geometric mean of active weights
        safe_weights = torch.where(
            b.bool(),
            w.clamp(min=1e-8),
            torch.ones_like(w),
        )

        log_w = torch.log(safe_weights)
        log_sum = (log_w * b).sum(dim=-1)  # (batch, vocab_size)

        n_active = b.sum(dim=-1).clamp(min=1)  # (batch, 1)
        log_geomean = log_sum / n_active
        logits = torch.exp(log_geomean)

        return logits


class VM2Model(nn.Module):
    """Full VM2 model with dynamic dendrite growth.

    Before feeding a longer input, call ensure_capacity(length)
    to grow the scoring layer if needed.
    """

    def __init__(self, config: VM2Config):
        super().__init__()
        self.config = config
        self.layer1 = BinaryDendriteLayer(config.vocab_size)
        self.layer2 = ScoringLayer(config, initial_positions=config.start_len)

    def ensure_capacity(self, seq_len: int):
        """Grow the model if seq_len exceeds current capacity."""
        if seq_len > self.layer2.current_positions:
            old = self.layer2.current_positions
            self.layer2.grow(seq_len)
            return old, seq_len  # grew from old to new
        return None

    def forward(self, char_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_values: (batch, seq_len) — ANY length, integer char values

        Returns:
            logits: (batch, vocab_size)
        """
        binary = self.layer1(char_values)
        logits = self.layer2(binary)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def current_positions(self):
        return self.layer2.current_positions
