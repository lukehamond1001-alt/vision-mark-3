# Vision Mark 3

**A multiplicative dendritic neural network for character-level language modeling.**

No attention. No ReLU. No softmax. No normalization layers. No embeddings.

All computation is multiplicative — weights scale inputs, geometric mean normalizes.

---

## Architecture

```
Layer 1: Binary Dendrite Layer (0 parameters — deterministic)
  ├── 1 neuron with seq_len × vocab_size dendrites
  ├── Each dendrite = one (position, character) KEY
  ├── Fires 1.0 if character matches at that position, 0.0 otherwise
  └── Output: binary vector with exactly N ones (N = visible chars)

Layer 2: Scoring Layer (learnable)
  ├── vocab_size neurons (one per possible next character)
  ├── Each neuron has one weight per dendrite KEY
  ├── Weights bounded positive: max_weight × sigmoid(raw_param)
  ├── Geometric mean of active weights → logit
  └── Output: vocab_size logits → cross-entropy loss
```

### What's NOT in the model

- No attention mechanism
- No ReLU / GELU / any activation function
- No softmax (except in loss during training)
- No LayerNorm / BatchNorm
- No learned embeddings
- No positional encoding — position is structural (which dendrite fires)

### Design principles

- **Layer 1 is a key**: each dendrite represents a (position, character) pair — like a lookup key. No multiplication, just match or no match
- **Layer 2 weights attach to keys**: the weight encodes how important "character C at position P" is for predicting each next character
- **Progressive reveal training**: each sequence produces `seq_len - 1` training samples by revealing one token at a time, training the model to predict with any amount of context

### Encoding

Simple integer mapping: `a=1, b=2, ..., z=26, space=27`. Text is lowercased and stripped to letters + spaces only.

`max_weight` = highest character number (27).

---

## Training Results

Trained on **Simple English Wikipedia** (5M characters, letters + spaces only).

### Full-context model (seq_len=46, 33,534 params)

| Steps | Train Loss | Val Loss |
|---|---|---|
| 100 | 3.44 | — |
| 5,000 | 2.87 | 2.86 |
| 50,000 | 2.50 | 2.50 |
| 200,000 | 2.46 | 2.46 |

Random baseline: ln(27) ≈ 3.30. Model exceeds it by step ~1,200.

### Progressive reveal model (seq_len=46, 33,534 params)

| Steps | Train Loss | Val Loss |
|---|---|---|
| 500 | 3.79 | — |
| 5,000 | 3.18 | 3.39 |
| 50,000 | 2.80 | 2.82 |

Higher loss because each sample averages over 1–45 chars of visible context — harder task than always seeing 46.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train on Simple Wikipedia (downloads automatically)
python -m vm2s.train --wiki --out_dir checkpoints --max_steps 50000

# Train on any text file
python -m vm2s.train --data path/to/text.txt --out_dir checkpoints

# Generate text
python -m vm2s.generate --checkpoint checkpoints/best.pt --prompt "the "

# Interactive mode
python -m vm2s.generate --checkpoint checkpoints/best.pt --interactive
```

### Training options

```bash
python -m vm2s.train \
    --wiki \
    --out_dir checkpoints \
    --seq_len 46 \
    --batch_size 32 \
    --lr 3e-4 \
    --max_steps 50000 \
    --log_interval 500 \
    --save_interval 5000
```

---

## Project Structure

```
vm2s/
  __init__.py      - Package init
  config.py        - VM2Config dataclass
  model.py         - BinaryDendriteLayer + ScoringLayer + VM2Model
  data.py          - Character encoding, Wikipedia download, progressive reveal dataset
  train.py         - Training loop with AdamW
  generate.py      - Autoregressive text generation
```

---

## How It Works

### Layer 1: Binary Dendrites

For input `"the cat"` with seq_len=10:

```
Position:  0  1  2  3  4  5  6  7  8  9
Input:     t  h  e     c  a  t  _  _  _
                                ↑ masked (0)

Dendrite for (pos=0, char='t') → 1  ✓ fires
Dendrite for (pos=0, char='a') → 0  ✗ doesn't fire
Dendrite for (pos=1, char='h') → 1  ✓ fires
Dendrite for (pos=7, char='a') → 0  ✗ masked position
...
```

Output: 1,242-dim binary vector (for seq_len=46, vocab=27).

### Layer 2: Scoring via Geometric Mean

Each of 27 output neurons has 1,242 weights. For each input:

1. Identify which dendrites fired (the binary 1s from Layer 1)
2. Gather corresponding weights
3. Compute geometric mean: `exp(mean(log(active_weights)))`
4. This becomes the logit for that output character

The weight at index `(pos × 27 + char)` encodes: *"How important is it that character C appeared at position P for predicting next character X?"*

---

## License

MIT
