"""Render scene-derived approximate CLEVR boxes for visual inspection."""

import argparse
from pathlib import Path

from spinls.clevr import CLEVR
from spinls.perception import clevr_scene_to_oracle_regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--scene", type=int, required=True, help="CLEVR image index")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from PIL import Image, ImageDraw

    dataset = CLEVR(args.data_dir, split=args.split)
    if args.scene not in dataset.scenes:
        raise SystemExit(f"Unknown {args.split} scene index: {args.scene}")
    scene = dataset.scenes[args.scene]
    image = Image.open(dataset.get_image_path(scene)).convert("RGB")
    regions = clevr_scene_to_oracle_regions(scene, image_size=image.size)
    draw = ImageDraw.Draw(image)
    for region, obj in zip(regions, scene["objects"], strict=True):
        draw.rectangle(region.box, outline="red", width=2)
        description = " ".join(
            str(obj[label]) for label in ("size", "color", "material", "shape")
        )
        x1, y1, _, _ = region.box
        draw.text(
            (x1 + 2, y1 + 2), f"{region.object_id} {description}",
            fill="white", stroke_width=2, stroke_fill="black",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Wrote {len(regions)} approximate boxes to {args.output}")


if __name__ == "__main__":
    main()
