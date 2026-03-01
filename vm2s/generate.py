"""
Text generation from a trained VM2 Simple checkpoint.

Usage:
    python -m vm2s.generate --checkpoint checkpoints/best.pt --prompt "the "
    python -m vm2s.generate --checkpoint checkpoints/best.pt --interactive
"""

import argparse
import torch
import torch.nn.functional as F

from vm2s.model import VM2Model, text_to_values, values_to_text, CHAR_MAP


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def generate(
    model: VM2Model,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text autoregressively."""
    model.eval()
    seq_len = model.config.seq_len

    # Clean prompt: lowercase, keep only known chars
    prompt = prompt.lower()
    values = text_to_values(prompt)

    for _ in range(max_tokens):
        # Take last seq_len values, pad with space if needed
        window = values[-seq_len:]
        while len(window) < seq_len:
            window = [CHAR_MAP[" "]] + window  # pad with space

        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x)
        logits = logits[0]

        if temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_vals[-1]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()
        next_val = next_idx + 1  # back to 1-indexed

        values.append(next_val)

    return values_to_text(values)


def main():
    parser = argparse.ArgumentParser(description="Generate text with VM2 Simple")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="the ", help="Starting text")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max chars to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling (0=disabled)")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt mode")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = VM2Model(config).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: step {ckpt.get('step', '?')}, params: {model.count_parameters():,}")

    if args.interactive:
        print("\nInteractive mode. Type a prompt, press Enter. Ctrl+C to quit.\n")
        while True:
            try:
                prompt = input(">>> ")
                if not prompt:
                    continue
                text = generate(model, prompt, args.max_tokens, args.temperature, args.top_k, device)
                print(text)
                print()
            except KeyboardInterrupt:
                print("\nDone.")
                break
    else:
        text = generate(model, args.prompt, args.max_tokens, args.temperature, args.top_k, device)
        print(text)


if __name__ == "__main__":
    main()
