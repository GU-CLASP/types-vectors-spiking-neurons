"""Feature-cache and factorized-head training utilities for CLEVR vision."""

from copy import deepcopy
import json
from pathlib import Path
import random

from .clevr import ATTRIBUTES
from .perception import TorchAttributeClassifier, build_factorized_attribute_heads


CACHE_SCHEMA_VERSION = 1
ATTRIBUTE_NAMES = tuple(ATTRIBUTES)
SHARD_KEYS = {"features", "labels", "boxes", "image_indices", "object_indices"}


def cache_manifest(cache_dir):
    """Load and minimally validate a feature-cache manifest."""
    path = Path(cache_dir) / "manifest.json"
    with path.open() as source:
        manifest = json.load(source)
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported feature-cache schema in {path}")
    expected = {name: list(values) for name, values in ATTRIBUTES.items()}
    if manifest.get("attributes") != expected:
        raise ValueError(f"CLEVR attribute vocabulary mismatch in {path}")
    feature_dim = manifest.get("feature_dim")
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError(f"Invalid feature dimension in {path}")
    return manifest


def feature_shards(cache_dir):
    """Return the ordered shard paths for a complete or partial cache."""
    paths = sorted(Path(cache_dir).glob("shard-*.pt"))
    if not paths:
        raise ValueError(f"No feature shards found in {cache_dir}")
    return paths


def write_feature_shard(path, scene_batches):
    """Atomically write aligned feature vectors, boxes, IDs, and class labels."""
    import torch

    features = []
    labels = []
    boxes = []
    image_indices = []
    object_indices = []
    for scene, objects in scene_batches:
        if len(objects) != len(scene["objects"]):
            raise ValueError("Each scene object must have exactly one feature vector")
        for index, (feature_object, obj) in enumerate(
            zip(objects, scene["objects"], strict=True)
        ):
            expected_id = f'{scene["image_index"]}-{index}'
            if feature_object.object_id != expected_id:
                raise ValueError("Feature objects are not identity-aligned with the scene")
            features.append(feature_object.features)
            labels.append([ATTRIBUTES[name].index(obj[name]) for name in ATTRIBUTE_NAMES])
            boxes.append(feature_object.box)
            image_indices.append(scene["image_index"])
            object_indices.append(index)
    if not features:
        raise ValueError("Cannot write an empty feature shard")

    payload = {
        "features": torch.tensor(features, dtype=torch.float16),
        "labels": torch.tensor(labels, dtype=torch.uint8),
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "image_indices": torch.tensor(image_indices, dtype=torch.int32),
        "object_indices": torch.tensor(object_indices, dtype=torch.int16),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return tuple(payload["features"].shape)


def load_feature_shard(path, *, feature_dim=None):
    """Load a tensor-only shard and validate its alignment and shape."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != SHARD_KEYS:
        raise ValueError(f"Invalid feature shard fields in {path}")
    rows = payload["features"].shape[0]
    if payload["features"].ndim != 2 or not rows:
        raise ValueError(f"Invalid feature matrix in {path}")
    if feature_dim is not None and payload["features"].shape[1] != feature_dim:
        raise ValueError(f"Feature dimension mismatch in {path}")
    expected_shapes = {
        "labels": (rows, len(ATTRIBUTE_NAMES)),
        "boxes": (rows, 4),
        "image_indices": (rows,),
        "object_indices": (rows,),
    }
    if any(tuple(payload[name].shape) != shape for name, shape in expected_shapes.items()):
        raise ValueError(f"Misaligned feature shard tensors in {path}")
    if not torch.isfinite(payload["features"]).all():
        raise ValueError(f"Non-finite features in {path}")
    return payload


def evaluate_attribute_heads(heads, cache_dir, *, batch_size=2048, device="cpu"):
    """Return per-family and macro top-1 accuracy over a feature cache."""
    import torch

    manifest = cache_manifest(cache_dir)
    correct = {name: 0 for name in ATTRIBUTE_NAMES}
    count = 0
    heads.eval()
    with torch.inference_mode():
        for path in feature_shards(cache_dir):
            payload = load_feature_shard(path, feature_dim=manifest["feature_dim"])
            for start in range(0, len(payload["features"]), batch_size):
                features = payload["features"][start:start + batch_size].to(
                    device=device, dtype=torch.float32
                )
                labels = payload["labels"][start:start + batch_size].to(
                    device=device, dtype=torch.long
                )
                for column, name in enumerate(ATTRIBUTE_NAMES):
                    predictions = heads[name](features).argmax(dim=1)
                    correct[name] += int((predictions == labels[:, column]).sum())
                count += len(features)
    accuracies = {name: correct[name] / count for name in ATTRIBUTE_NAMES}
    accuracies["macro"] = sum(accuracies.values()) / len(ATTRIBUTE_NAMES)
    accuracies["objects"] = count
    return accuracies


def train_attribute_heads(
    train_cache,
    val_cache,
    output,
    *,
    epochs=10,
    batch_size=1024,
    learning_rate=1e-3,
    weight_decay=1e-4,
    device="cpu",
    seed=0,
    progress=print,
):
    """Train linear attribute heads and atomically save the best validation model."""
    import torch
    from torch.nn import functional as F

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Invalid classifier-head training hyperparameters")
    train_manifest = cache_manifest(train_cache)
    val_manifest = cache_manifest(val_cache)
    if train_manifest["feature_dim"] != val_manifest["feature_dim"]:
        raise ValueError("Train and validation feature dimensions differ")
    if train_manifest.get("extractor") != val_manifest.get("extractor"):
        raise ValueError("Train and validation caches use different extractors")

    torch.manual_seed(seed)
    heads = build_factorized_attribute_heads(train_manifest["feature_dim"]).to(device)
    optimizer = torch.optim.AdamW(
        heads.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    shard_paths = feature_shards(train_cache)
    best_accuracy = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        heads.train()
        epoch_loss = 0.0
        object_count = 0
        shuffled_paths = list(shard_paths)
        random.Random(seed + epoch).shuffle(shuffled_paths)
        for path in shuffled_paths:
            payload = load_feature_shard(path, feature_dim=train_manifest["feature_dim"])
            order = torch.randperm(len(payload["features"]), generator=generator)
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                features = payload["features"][indices].to(
                    device=device, dtype=torch.float32
                )
                labels = payload["labels"][indices].to(device=device, dtype=torch.long)
                loss = sum(
                    F.cross_entropy(heads[name](features), labels[:, column])
                    for column, name in enumerate(ATTRIBUTE_NAMES)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(features)
                object_count += len(features)

        metrics = evaluate_attribute_heads(
            heads, val_cache, batch_size=batch_size, device=device
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = epoch_loss / object_count
        history.append(metrics)
        progress(
            f"epoch {epoch}: loss={metrics['train_loss']:.4f}, "
            + ", ".join(
                f"{name}={metrics[name]:.2%}" for name in (*ATTRIBUTE_NAMES, "macro")
            )
        )
        if metrics["macro"] > best_accuracy:
            best_accuracy = metrics["macro"]
            best_state = deepcopy(heads.state_dict())
            checkpoint = {
                "schema_version": 1,
                "feature_dim": train_manifest["feature_dim"],
                "attributes": {name: list(values) for name, values in ATTRIBUTES.items()},
                "extractor": train_manifest.get("extractor"),
                "model_state_dict": best_state,
                "best_epoch": epoch,
                "best_validation": dict(metrics),
                "history": list(history),
            }
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(output)

    heads.load_state_dict(best_state)
    return heads, history


def load_attribute_classifier(checkpoint_path, *, device="cpu"):
    """Load a trained head checkpoint as a feature-distribution callback."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = {name: list(values) for name, values in ATTRIBUTES.items()}
    if checkpoint.get("schema_version") != 1 or checkpoint.get("attributes") != expected:
        raise ValueError("Unsupported attribute-head checkpoint")
    heads = build_factorized_attribute_heads(checkpoint["feature_dim"])
    heads.load_state_dict(checkpoint["model_state_dict"])
    return TorchAttributeClassifier(heads, device=device)
