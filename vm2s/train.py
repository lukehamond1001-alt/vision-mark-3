"""
Training loop for VM2 Simple — with dynamic dendrite growth.

The model starts with short sequences and grows its dendrites
as training progresses, handling longer and longer inputs.

Usage:
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
        max_weight=VOCAB_SIZE,
        start_len=args.start_len,
        grow_every=args.grow_every,
        max_len=args.max_len,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
    )

    # Model — starts with start_len positions
    model = VM2Model(config).to(device)
    current_seq_len = config.start_len
    print(f"Starting seq_len: {current_seq_len}")
    print(f"Initial params: {model.count_parameters():,}")
    print(f"Will grow by 1 position every {config.grow_every} steps up to {config.max_len}")

    # Optimizer — will be recreated after each grow
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Create initial dataloaders
    train_loader, val_loader = create_dataloaders(
        train_text, val_text, current_seq_len, config.batch_size,
    )

    # Training state
    step = 0
    best_val_loss = float("inf")
    log_interval = args.log_interval

    model.train()
    t0 = time.time()
    running_loss = 0.0
    tokens_seen = 0
    last_grow_step = 0

    for epoch in range(args.epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

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

            # Log
            if step > 0 and step % log_interval == 0 and (batch_idx + 1) % config.grad_accum_steps == 0:
                dt = time.time() - t0
                avg_loss = running_loss / log_interval
                tps = tokens_seen / dt
                print(
                    f"step {step:>6d} | len {current_seq_len:>3d} | "
                    f"params {model.count_parameters():>8,d} | "
                    f"loss {avg_loss:.4f} | {tps:,.0f} tok/s"
                )
                running_loss = 0.0
                tokens_seen = 0
                t0 = time.time()

            # GROW: increase sequence length
            if (step > 0 and step % config.grow_every == 0
                and current_seq_len < config.max_len
                and (batch_idx + 1) % config.grad_accum_steps == 0
                and step > last_grow_step):

                last_grow_step = step
                current_seq_len += 1

                # Grow model dendrites
                grew = model.ensure_capacity(current_seq_len)
                if grew:
                    # Recreate optimizer to include new parameters
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=config.lr,
                        weight_decay=config.weight_decay,
                    )
                    print(f"  >> GREW to seq_len={current_seq_len}, "
                          f"params={model.count_parameters():,}")

                # Recreate dataloaders for new length
                train_loader, val_loader = create_dataloaders(
                    train_text, val_text, current_seq_len, config.batch_size,
                )
                break  # break inner loop to restart with new dataloader

            # Save
            if step > 0 and step % args.save_interval == 0 and (batch_idx + 1) % config.grad_accum_steps == 0:
                _save_checkpoint(model, optimizer, config, step, current_seq_len, args.out_dir, "latest.pt")

                if val_loader is not None:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"  val_loss: {val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save_checkpoint(model, optimizer, config, step, current_seq_len, args.out_dir, "best.pt")
                        print(f"  new best!")
                    model.train()

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    _save_checkpoint(model, optimizer, config, step, current_seq_len, args.out_dir, "final.pt")
    print(f"Training complete. {step} steps. Final seq_len={current_seq_len}, params={model.count_parameters():,}")


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


def _save_checkpoint(model, optimizer, config, step, seq_len, out_dir, filename):
    path = os.path.join(out_dir, filename)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
        "seq_len": seq_len,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train VM2 Simple (growing)")
    parser.add_argument("--data", type=str, default=None, help="Path to training text file")
    parser.add_argument("--wiki", action="store_true", help="Download and use Simple Wikipedia")
    parser.add_argument("--max_chars", type=int, default=5_000_000, help="Max chars from Wikipedia")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Checkpoint output dir")
    parser.add_argument("--start_len", type=int, default=2, help="Starting sequence length")
    parser.add_argument("--grow_every", type=int, default=500, help="Grow seq_len by 1 every N steps")
    parser.add_argument("--max_len", type=int, default=256, help="Max sequence length to grow to")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Max epochs")
    parser.add_argument("--max_steps", type=int, default=200000, help="Max training steps")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    if args.data is None and not args.wiki:
        parser.error("Provide --data or --wiki")

    train(args)


if __name__ == "__main__":
    main()
