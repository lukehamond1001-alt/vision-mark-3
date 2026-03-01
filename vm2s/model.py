"""
VM2 Simple Model: 2-layer dendritic character-level predictor.

Layer 1 (BinaryDendriteLayer):
  - No learnable parameters. Deterministic.
  - 1 neuron with seq_len * vocab_size dendrites.
  - Each dendrite = one (position, character) key.
  - Output: binary vector — 1 if the character at that position matches, 0 otherwise.
  - Exactly seq_len dendrites fire per input (one per position).

Layer 2 (ScoringLayer):
  - vocab_size neurons (one per possible next character).
  - Each neuron has one weight per Layer 1 dendrite (per key).
  - Weights are bounded positive: max_weight * sigmoid(raw_param).
  - Output = geometric mean of the weights at active (fired) dendrites.
  - These outputs serve as logits for next-character prediction.
"""

import torch
import torch.nn as nn
from typing import Tuple

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

    Each dendrite is a (position, character) key.
    Dendrite index = position * vocab_size + (char_value - 1).

    Input:  (batch, seq_len) integer char values
    Output: (batch, seq_len * vocab_size) binary tensor
    """

    def __init__(self, config: VM2Config):
        super().__init__()
        self.seq_len = config.seq_len
        self.vocab_size = config.vocab_size
        self.num_dendrites = config.seq_len * config.vocab_size

    def forward(self, char_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_values: (batch, seq_len) — integer values 1..vocab_size, 0 = masked/hidden

        Returns:
            binary: (batch, num_dendrites) — fires only for revealed positions
        """
        batch = char_values.shape[0]

        # Create mask: 1 for revealed positions (value > 0), 0 for hidden
        revealed = (char_values > 0).float()  # (batch, seq_len)

        # One-hot encode each position: (batch, seq_len, vocab_size)
        # char_values are 1-indexed, shift to 0-indexed for one_hot
        # Clamp so masked positions (0) map to index 0 — we'll zero them out after
        zero_indexed = (char_values - 1).clamp(min=0)
        one_hot = torch.nn.functional.one_hot(zero_indexed, self.vocab_size).float()

        # Zero out masked positions: multiply by revealed mask
        # revealed: (batch, seq_len) → (batch, seq_len, 1)
        one_hot = one_hot * revealed.unsqueeze(-1)

        # Flatten to (batch, seq_len * vocab_size)
        binary = one_hot.reshape(batch, -1)

        return binary


class ScoringLayer(nn.Module):
    """Layer 2: learned scoring via multiplicative weights.

    Each of vocab_size neurons has one weight per Layer 1 dendrite.
    The weight is the learned value for that (position, character) key.
    Weights are bounded positive: max_weight * sigmoid(raw_param).

    For each input:
      - Identify which dendrites fired (binary = 1)
      - Gather the corresponding weights
      - Compute geometric mean of those weights → output logit

    Output: (batch, vocab_size) logits
    """

    def __init__(self, config: VM2Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.num_dendrites = config.seq_len * config.vocab_size
        self.max_weight = float(config.max_weight)

        # Raw weight parameters: (vocab_size, num_dendrites)
        # Each row = one output neuron's weights over all (pos, char) keys
        self.raw_weights = nn.Parameter(
            torch.zeros(self.vocab_size, self.num_dendrites)
        )
        nn.init.uniform_(self.raw_weights, -1.0, 1.0)

    def _get_weights(self) -> torch.Tensor:
        """Compute bounded positive weights from raw parameters."""
        return self.max_weight * torch.sigmoid(self.raw_weights)

    def forward(self, binary: torch.Tensor) -> torch.Tensor:
        """
        Args:
            binary: (batch, num_dendrites) — binary activation from Layer 1

        Returns:
            logits: (batch, vocab_size)
        """
        weights = self._get_weights()  # (vocab_size, num_dendrites)

        # Number of active dendrites per sample (should be seq_len)
        # binary shape: (batch, num_dendrites)
        # weights shape: (vocab_size, num_dendrites)

        # Expand for broadcasting:
        # binary: (batch, 1, num_dendrites)
        # weights: (1, vocab_size, num_dendrites)
        b = binary.unsqueeze(1)       # (batch, 1, num_dendrites)
        w = weights.unsqueeze(0)      # (1, vocab_size, num_dendrites)

        # Mask: only active dendrites contribute
        # For inactive dendrites, substitute 1.0 so log(1)=0 (neutral in product)
        safe_weights = torch.where(
            b.bool(),
            w.clamp(min=1e-8),
            torch.ones_like(w),
        )

        # Geometric mean via log-space:
        # log of each active weight, sum, divide by n, exp
        log_w = torch.log(safe_weights)
        log_sum = (log_w * b).sum(dim=-1)  # (batch, vocab_size)

        n_active = b.sum(dim=-1)  # (batch, 1) — should be seq_len
        safe_n = n_active.clamp(min=1)

        log_geomean = log_sum / safe_n
        logits = torch.exp(log_geomean)  # (batch, vocab_size)

        return logits


class VM2Model(nn.Module):
    """Full VM2 Simple model: Layer 1 (binary) → Layer 2 (scoring).

    Layer 1 asks: "what character is at each position?"
    Layer 2 asks: "given what's where, what comes next?"
    """

    def __init__(self, config: VM2Config):
        super().__init__()
        self.config = config
        self.layer1 = BinaryDendriteLayer(config)
        self.layer2 = ScoringLayer(config)

    def forward(self, char_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_values: (batch, seq_len) — integer char values 1..vocab_size

        Returns:
            logits: (batch, vocab_size)
        """
        binary = self.layer1(char_values)
        logits = self.layer2(binary)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
