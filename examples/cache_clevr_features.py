"""Cache Faster R-CNN box-head vectors for CLEVR pseudo-oracle regions."""

import argparse
import json
from pathlib import Path

from spinls.clevr import ATTRIBUTES
from spinls.perception import (
    FasterRCNNFeatureExtractor,
    clevr_scene_to_oracle_regions,
)
from spinls.vision_training import CACHE_SCHEMA_VERSION, load_feature_shard, write_feature_shard


def write_manifest(path, manifest):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2, help="Images per backbone pass")
    parser.add_argument("--scenes-per-shard", type=int, default=100)
    parser.add_argument("--start-scene", type=int, default=0, help="First scene-list offset")
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument(
        "--architecture",
        choices=("mobilenet-v3-320-fpn", "resnet50-fpn"),
        default="mobilenet-v3-320-fpn",
        help="Torchvision Faster R-CNN backbone (MobileNet is much faster on CPU)",
    )
    parser.add_argument(
        "--weights", choices=("coco", "none"), default="coco",
        help="Use COCO pretrained weights; 'none' is only for pipeline tests",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.scenes_per_shard <= 0 or args.start_scene < 0:
        parser.error("batch size and shard size must be positive; start must be nonnegative")
    if args.limit_scenes is not None and args.limit_scenes <= 0:
        parser.error("--limit-scenes must be positive")

    scene_path = args.data_dir / "scenes" / f"CLEVR_{args.split}_scenes.json"
    with scene_path.open() as source:
        all_scenes = json.load(source)["scenes"]
    stop = len(all_scenes) if args.limit_scenes is None else args.start_scene + args.limit_scenes
    stop = min(stop, len(all_scenes))
    scenes = all_scenes[args.start_scene:stop]
    if not scenes:
        parser.error("the requested scene range is empty")

    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
        fasterrcnn_resnet50_fpn,
    )

    architectures = {
        "mobilenet-v3-320-fpn": (
            fasterrcnn_mobilenet_v3_large_320_fpn,
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
        ),
        "resnet50-fpn": (
            fasterrcnn_resnet50_fpn,
            FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        ),
    }
    constructor, default_weights = architectures[args.architecture]
    if args.weights == "coco":
        weights = default_weights
        model = constructor(weights=weights)
        weight_name = weights.name
    else:
        model = constructor(weights=None, weights_backbone=None)
        weight_name = None
        print("WARNING: random weights are not meaningful perceptual features")
    extractor_metadata = {
        "architecture": args.architecture,
        "weights": weight_name,
    }

    import torch
    import torchvision
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    extractor_metadata["torch"] = torch.__version__
    extractor_metadata["torchvision"] = torchvision.__version__
    requested_manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "split": args.split,
        "scene_start": args.start_scene,
        "scene_stop": stop,
        "scenes_per_shard": args.scenes_per_shard,
        "feature_dtype": "float16",
        "attributes": {name: list(values) for name, values in ATTRIBUTES.items()},
        "extractor": extractor_metadata,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        comparable = dict(existing)
        comparable.pop("feature_dim", None)
        if comparable != requested_manifest:
            raise SystemExit(f"Existing cache configuration differs: {manifest_path}")

    extractor = FasterRCNNFeatureExtractor(model, device=args.device)
    feature_dim = None
    for shard_start in range(0, len(scenes), args.scenes_per_shard):
        shard_scenes = scenes[shard_start:shard_start + args.scenes_per_shard]
        first = shard_scenes[0]["image_index"]
        last = shard_scenes[-1]["image_index"]
        shard_path = args.output / f"shard-{first:06d}-{last:06d}.pt"
        if shard_path.exists():
            payload = load_feature_shard(shard_path, feature_dim=feature_dim)
            feature_dim = payload["features"].shape[1]
            print(f"Skipping complete {shard_path.name}")
            continue

        completed = []
        for batch_start in range(0, len(shard_scenes), args.batch_size):
            batch = shard_scenes[batch_start:batch_start + args.batch_size]
            images = []
            regions = []
            for scene in batch:
                image_path = args.data_dir / "images" / args.split / scene["image_filename"]
                with Image.open(image_path) as source:
                    image = pil_to_tensor(source.convert("RGB")).float().div_(255)
                images.append(image)
                regions.append(clevr_scene_to_oracle_regions(scene, image_size=(image.shape[2], image.shape[1])))
            feature_batches = extractor.extract_batch(images, regions)
            completed.extend(zip(batch, feature_batches, strict=True))

        rows, current_dim = write_feature_shard(shard_path, completed)
        if feature_dim is not None and current_dim != feature_dim:
            raise ValueError("Extractor feature dimension changed during export")
        feature_dim = current_dim
        manifest = dict(requested_manifest, feature_dim=feature_dim)
        write_manifest(manifest_path, manifest)
        print(f"Wrote {shard_path.name}: {rows} objects x {feature_dim} features")

    if not manifest_path.exists():
        write_manifest(manifest_path, dict(requested_manifest, feature_dim=feature_dim))
    print(f"Feature cache complete: {args.output}")


if __name__ == "__main__":
    main()
