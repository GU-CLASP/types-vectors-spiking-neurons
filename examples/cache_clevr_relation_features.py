"""Cache ordered-pair CLEVR relation features from ROI and scene vectors."""

import argparse
import json
from pathlib import Path

from spinls.perception import FasterRCNNFeatureExtractor
from spinls.relation_training import (
    CANONICAL_RELATIONS,
    RELATION_SCHEMA_VERSION,
    load_relation_shard,
    write_relation_shard,
)
from spinls.vision_training import cache_manifest, feature_shards, load_feature_shard


def write_manifest(path, manifest):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def scenes_by_index(data_dir, split):
    with (data_dir / "scenes" / f"CLEVR_{split}_scenes.json").open() as source:
        return {
            scene["image_index"]: scene
            for scene in json.load(source)["scenes"]
        }


def grouped_object_features(payload):
    groups = {}
    for row, (image_index, object_index) in enumerate(
        zip(payload["image_indices"].tolist(), payload["object_indices"].tolist(), strict=True)
    ):
        groups.setdefault(int(image_index), {})[int(object_index)] = payload["features"][row]
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--object-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8, help="Images per backbone pass")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    manifest = cache_manifest(args.object_cache)
    extractor_manifest = manifest.get("extractor") or {}
    architecture = extractor_manifest.get("architecture")
    weights_name = extractor_manifest.get("weights")
    if architecture != "resnet50-fpn" or weights_name not in ("COCO_V1", "DEFAULT"):
        parser.error("relation scene features currently require COCO ResNet50-FPN object features")

    from PIL import Image
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
    from torchvision.transforms.functional import pil_to_tensor

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    extractor = FasterRCNNFeatureExtractor(model, device=args.device)
    scenes = scenes_by_index(args.data_dir, args.split)

    requested_manifest = {
        "schema_version": RELATION_SCHEMA_VERSION,
        "split": args.split,
        "relations": list(CANONICAL_RELATIONS),
        "object_feature_dim": manifest["feature_dim"],
        "scene_feature_dtype": "float16",
        "pair_feature_dtype": "float16",
        "extractor": extractor_manifest,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        comparable = dict(existing)
        comparable.pop("scene_feature_dim", None)
        comparable.pop("feature_dim", None)
        if comparable != requested_manifest:
            raise SystemExit(f"Existing relation cache configuration differs: {manifest_path}")

    scene_feature_dim = None
    relation_feature_dim = None
    for object_shard in feature_shards(args.object_cache):
        relation_shard = args.output / object_shard.name
        if relation_shard.exists():
            payload = load_relation_shard(relation_shard, feature_dim=relation_feature_dim)
            relation_feature_dim = payload["features"].shape[1]
            scene_feature_dim = relation_feature_dim - 2 * manifest["feature_dim"]
            print(f"Skipping complete {relation_shard.name}")
            continue

        object_payload = load_feature_shard(object_shard, feature_dim=manifest["feature_dim"])
        grouped = grouped_object_features(object_payload)
        ordered_scene_indices = sorted(grouped)
        completed = []
        for start in range(0, len(ordered_scene_indices), args.batch_size):
            batch_indices = ordered_scene_indices[start:start + args.batch_size]
            images = []
            batch_scenes = []
            for image_index in batch_indices:
                scene = scenes[image_index]
                image_path = args.data_dir / "images" / args.split / scene["image_filename"]
                with Image.open(image_path) as source:
                    image = pil_to_tensor(source.convert("RGB")).float().div_(255)
                images.append(image)
                batch_scenes.append(scene)
            scene_vectors = extractor.extract_scene_features_batch(images)
            for scene, scene_features in zip(batch_scenes, scene_vectors, strict=True):
                by_object = grouped[scene["image_index"]]
                object_features = [
                    tuple(float(value) for value in by_object[index])
                    for index in range(len(scene["objects"]))
                ]
                completed.append((scene, object_features, scene_features))

        rows, current_dim = write_relation_shard(relation_shard, completed)
        if relation_feature_dim is not None and current_dim != relation_feature_dim:
            raise ValueError("Relation feature dimension changed during export")
        relation_feature_dim = current_dim
        scene_feature_dim = relation_feature_dim - 2 * manifest["feature_dim"]
        write_manifest(manifest_path, dict(
            requested_manifest,
            scene_feature_dim=scene_feature_dim,
            feature_dim=relation_feature_dim,
        ))
        print(f"Wrote {relation_shard.name}: {rows} pairs x {relation_feature_dim} features")

    if not manifest_path.exists():
        write_manifest(manifest_path, dict(
            requested_manifest,
            scene_feature_dim=scene_feature_dim,
            feature_dim=relation_feature_dim,
        ))
    print(f"Relation feature cache complete: {args.output}")


if __name__ == "__main__":
    main()
