"""Render image-only CLEVR detector boxes and scene-local object IDs."""

import argparse
from pathlib import Path

from spinls.detection_training import detect_entities, load_clevr_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--image-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.image_index < 0:
        parser.error("--image-index must be nonnegative")

    from PIL import Image, ImageDraw
    from torchvision.transforms.functional import pil_to_tensor

    filename = f"CLEVR_{args.split}_{args.image_index:06d}.png"
    image_path = args.data_dir / "images" / args.split / filename
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tensor = pil_to_tensor(image).float().div_(255)
    model = load_clevr_detector(args.checkpoint, device=args.device)
    detections = detect_entities(
        model,
        tensor,
        image_id=str(args.image_index),
        score_threshold=args.score_threshold,
        device=args.device,
    )

    draw = ImageDraw.Draw(image)
    for detection in detections:
        draw.rectangle(detection.box, outline="red", width=2)
        x1, y1, _, _ = detection.box
        draw.text(
            (x1 + 2, y1 + 2),
            f"{detection.object_id} {detection.score:.2f}",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Wrote {len(detections)} image-only detections to {args.output}")


if __name__ == "__main__":
    main()
