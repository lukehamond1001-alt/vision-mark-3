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

## Architecture Update: Dynamic Dendrite Growth

In Vision Mark 2, `seq_len` was fixed at 46. Vision Mark 3 removes this restriction: **the model dynamically grows its dendrites to match the input length.**

- **Starts small**: Training begins at `seq_len = 2`.
- **Grows on demand**: Every 500 steps, if a longer sequence is needed, Layer 2 dynamically adds randomly initialized weights for the new positions.
- **Permanent capacity**: Once grown, the new dendrites are permanently available. Shorter inputs simply leave the later dendrites inactive (zeros).

---

## Training Results

Trained on **Simple English Wikipedia** (5M characters, letters + spaces only).

### Dynamic Growth Model (100K steps)

- **Starting parameters**: 1,458 (`seq_len=2`)
- **Final parameters**: 147,258 (`seq_len=202`)
- **Final loss**: 2.89

| Steps | seq_len | Params | Loss |
|---|---|---|---|
| 0 | 2 | 1,458 | 3.29 |
| 10,000 | 22 | 16,038 | 2.97 |
| 22,000 | 46 | 33,534 | 2.88 |
| 50,000 | 102 | 74,358 | 2.88 |
| 100,000 | 202 | 147,258 | 2.89 |

### The "Signal Dilution" Bottleneck

While the model successfully learned to handle variable-length sequences up to 202 characters, text generation quality at long contexts degrades compared to the fixed 46-char model.

**Prompt**: `"the cat "`
**Output (202-char model)**: `"the cat ee te te a at a atat t eee tehete eate ethe te ewe e te ate ..."`

**Why?**
Because Layer 2 uses a pure geometric mean across ALL active dendrites. 
- With a 46-char context, each character casts a $\frac{1}{46}$ vote.
- With a 202-char context, each character casts a $\frac{1}{202}$ vote.

Even if the model learns that the 3 most recent characters are critical for predicting the next one, their strong signal is completely washed out by the 199 neutral/noisy signals from the earlier context window. This proves that a purely multiplicative architecture *must* have a mechanism to dynamically weight or ignore irrelevant past context (e.g. self-attention or a hidden layer) to support long sequences.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train with dynamic growth (starts at len 2, grows up to 256)
python -m vm2s.train --wiki --out_dir checkpoints \
    --start_len 2 --grow_every 500 --max_len 256 --max_steps 100000

# Generate text (model grows dynamically if prompt exceeds capacity)
python -m vm2s.generate --checkpoint checkpoints/final.pt --prompt "the "

# Interactive mode
python -m vm2s.generate --checkpoint checkpoints/final.pt --interactive
```

### Training options

```bash
python -m vm2s.train \
    --data path/to/text.txt \
    --out_dir checkpoints \
    --start_len 2 \
    --grow_every 500 \
    --max_len 256 \
    --batch_size 32 \
    --lr 3e-4 \
    --max_steps 100000 \
    --log_interval 200 \
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
