"""
Training loop for VM2 Simple.

Usage:
    python -m vm2s.train --data path/to/text.txt --out_dir checkpoints
    python -m vm2s.train --wiki --out_dir checkpoints
"""

import os
import time
import argparse
import torch
import torch.nn.functional as F

from vm2s.config import VM2Config
from vm2s.model import VM2Model, VOCAB_SIZE
from vm2s.data import load_text, create_dataloaders, download_simple_wikipedia


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # Load data
    if args.wiki:
        data_path = os.path.join(args.out_dir, "simple_wiki.txt")
        download_simple_wikipedia(data_path, max_chars=args.max_chars)
        text = load_text(data_path)
    else:
        text = load_text(args.data)

    split = int(len(text) * 0.9)
    train_text = text[:split]
    val_text = text[split:]
    print(f"Train chars: {len(train_text):,}  Val chars: {len(val_text):,}")

    # Config
    config = VM2Config(
        vocab_size=VOCAB_SIZE,
        seq_len=args.seq_len,
        max_weight=VOCAB_SIZE,  # highest char number
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
    )

    # Dataloaders
    train_loader, val_loader = create_dataloaders(
        train_text, val_text, config.seq_len, config.batch_size,
    )

    # Model
    model = VM2Model(config).to(device)
    num_params = model.count_parameters()
    num_dendrites = config.seq_len * config.vocab_size
    print(f"Parameters: {num_params:,}")
    print(f"Layer 1: {num_dendrites} dendrites (no weights)")
    print(f"Layer 2: {config.vocab_size} neurons × {num_dendrites} weights")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Resume
    start_step = 0
    if args.resume and os.path.exists(os.path.join(args.out_dir, "latest.pt")):
        ckpt = torch.load(os.path.join(args.out_dir, "latest.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")
        del ckpt

    # Training loop
    step = start_step
    best_val_loss = float("inf")
    log_interval = args.log_interval

    model.train()
    t0 = time.time()
    running_loss = 0.0
    tokens_seen = 0

    for epoch in range(args.epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            # Target: y is 1-indexed char value, need 0-indexed for cross-entropy
            target = (y - 1).clamp(min=0)

            logits = model(x)
            loss = F.cross_entropy(logits, target)

            loss = loss / config.grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

            running_loss += loss.item() * config.grad_accum_steps
            tokens_seen += x.shape[0] * x.shape[1]

            if step > 0 and step % log_interval == 0 and (batch_idx + 1) % config.grad_accum_steps == 0:
                dt = time.time() - t0
                avg_loss = running_loss / log_interval
                tps = tokens_seen / dt
                print(
                    f"step {step:>6d} | loss {avg_loss:.4f} | "
                    f"{tps:,.0f} tok/s | {dt:.1f}s"
                )
                running_loss = 0.0
                tokens_seen = 0
                t0 = time.time()

            if step > 0 and step % args.save_interval == 0 and (batch_idx + 1) % config.grad_accum_steps == 0:
                _save_checkpoint(model, optimizer, config, step, args.out_dir, "latest.pt")

                if val_loader is not None:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"  val_loss: {val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save_checkpoint(model, optimizer, config, step, args.out_dir, "best.pt")
                        print(f"  new best!")
                    model.train()

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    _save_checkpoint(model, optimizer, config, step, args.out_dir, "final.pt")
    print(f"Training complete. {step} steps.")


def evaluate(model, val_loader, device, max_batches=200):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            target = (y - 1).clamp(min=0)
            logits = model(x)
            loss = F.cross_entropy(logits, target)
            total_loss += loss.item()
            count += 1
            if count >= max_batches:
                break
    return total_loss / max(count, 1)


def _save_checkpoint(model, optimizer, config, step, out_dir, filename):
    path = os.path.join(out_dir, filename)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train VM2 Simple")
    parser.add_argument("--data", type=str, default=None, help="Path to training text file")
    parser.add_argument("--wiki", action="store_true", help="Download and use Simple Wikipedia")
    parser.add_argument("--max_chars", type=int, default=5_000_000, help="Max chars from Wikipedia")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Checkpoint output dir")
    parser.add_argument("--seq_len", type=int, default=46, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--max_steps", type=int, default=100000, help="Max training steps")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--log_interval", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    if args.data is None and not args.wiki:
        parser.error("Provide --data or --wiki")

    train(args)


if __name__ == "__main__":
    main()
