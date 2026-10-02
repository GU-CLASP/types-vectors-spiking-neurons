"""Evaluate a trained CLEVR object detector against approximate validation boxes."""

import argparse
from pathlib import Path

from spinls.detection_training import (
    CLEVRDetectionDataset,
    detection_collate,
    evaluate_detector,
    load_clevr_detector,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from torch.utils.data import DataLoader

    data = CLEVRDetectionDataset(args.data_dir, args.split, limit=args.limit_scenes)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=detection_collate,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    model = load_clevr_detector(args.checkpoint, device=args.device)
    metrics = evaluate_detector(
        model,
        loader,
        device=args.device,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
