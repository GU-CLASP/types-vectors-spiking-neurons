"""Train factorized CLEVR attribute heads from cached box-head vectors."""

import argparse
from pathlib import Path

from spinls.vision_training import train_attribute_heads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    _, history = train_attribute_heads(
        args.train_cache,
        args.val_cache,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
    )
    best = max(history, key=lambda metrics: metrics["macro"])
    print(
        f"Saved {args.output}; best epoch {best['epoch']} "
        f"validation macro accuracy={best['macro']:.2%}"
    )


if __name__ == "__main__":
    main()
