"""Pair-feature caches and binary relation-head training for CLEVR."""

from copy import deepcopy
import json
from pathlib import Path
import random


RELATION_SCHEMA_VERSION = 1
CANONICAL_RELATIONS = ("right", "front")
RELATION_SHARD_KEYS = {
    "features",
    "labels",
    "image_indices",
    "target_indices",
    "reference_indices",
}


def build_relation_heads(input_dim):
    """Build one binary linear head for each canonical spatial relation."""
    if not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError("input_dim must be a positive integer")
    from torch import nn

    return nn.ModuleDict({
        relation: nn.Linear(input_dim, 2)
        for relation in CANONICAL_RELATIONS
    })


class TorchRelationClassifier:
    """Adapt trained relation heads to ordered pair -> probability maps."""

    def __init__(self, heads, *, device="cpu"):
        if set(heads) != set(CANONICAL_RELATIONS):
            raise ValueError("Relation heads must cover right and front")
        self.heads = heads.to(device).eval()
        self.device = device

    def __call__(self, target_features, reference_features, scene_features):
        import torch

        vector = torch.tensor(
            tuple(target_features) + tuple(reference_features) + tuple(scene_features),
            dtype=torch.float32,
            device=self.device,
        )
        result = {}
        with torch.inference_mode():
            for relation in CANONICAL_RELATIONS:
                probabilities = self.heads[relation](vector).softmax(dim=-1).cpu()
                result[relation] = {
                    False: float(probabilities[0]),
                    True: float(probabilities[1]),
                }
        return result


def relation_cache_manifest(cache_dir):
    """Load and validate a relation-feature cache manifest."""
    path = Path(cache_dir) / "manifest.json"
    with path.open() as source:
        manifest = json.load(source)
    if manifest.get("schema_version") != RELATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported relation-cache schema in {path}")
    for key in ("object_feature_dim", "scene_feature_dim", "feature_dim"):
        value = manifest.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Invalid {key} in {path}")
    expected = 2 * manifest["object_feature_dim"] + manifest["scene_feature_dim"]
    if manifest["feature_dim"] != expected:
        raise ValueError(f"Relation feature dimension mismatch in {path}")
    if tuple(manifest.get("relations", ())) != CANONICAL_RELATIONS:
        raise ValueError(f"Relation vocabulary mismatch in {path}")
    return manifest


def relation_shards(cache_dir):
    """Return ordered relation-cache shard paths."""
    paths = sorted(Path(cache_dir).glob("shard-*.pt"))
    if not paths:
        raise ValueError(f"No relation shards found in {cache_dir}")
    return paths


def write_relation_shard(path, scene_batches):
    """Write all ordered non-self object pairs for each scene."""
    import torch

    features = []
    labels = []
    image_indices = []
    target_indices = []
    reference_indices = []
    object_dim = None
    scene_dim = None
    for scene, object_features, scene_features in scene_batches:
        objects = scene["objects"]
        object_features = [tuple(feature) for feature in object_features]
        scene_features = tuple(scene_features)
        if len(object_features) != len(objects):
            raise ValueError("Each scene object must have exactly one feature vector")
        if object_dim is None:
            object_dim = len(object_features[0]) if object_features else None
            scene_dim = len(scene_features)
        if not object_dim or not scene_dim:
            raise ValueError("Object and scene feature vectors must be nonempty")
        if any(len(feature) != object_dim for feature in object_features):
            raise ValueError("Object feature dimensions changed within relation shard")
        if len(scene_features) != scene_dim:
            raise ValueError("Scene feature dimensions changed within relation shard")
        relationships = scene.get("relationships", {})
        for relation in CANONICAL_RELATIONS:
            if len(relationships.get(relation, ())) != len(objects):
                raise ValueError(f"Scene is missing {relation} relationships")
        for reference_index in range(len(objects)):
            positives = {
                relation: set(relationships[relation][reference_index])
                for relation in CANONICAL_RELATIONS
            }
            for target_index in range(len(objects)):
                if target_index == reference_index:
                    continue
                features.append(
                    object_features[target_index]
                    + object_features[reference_index]
                    + scene_features
                )
                labels.append([
                    int(target_index in positives[relation])
                    for relation in CANONICAL_RELATIONS
                ])
                image_indices.append(scene["image_index"])
                target_indices.append(target_index)
                reference_indices.append(reference_index)
    if not features:
        raise ValueError("Cannot write an empty relation shard")

    payload = {
        "features": torch.tensor(features, dtype=torch.float16),
        "labels": torch.tensor(labels, dtype=torch.uint8),
        "image_indices": torch.tensor(image_indices, dtype=torch.int32),
        "target_indices": torch.tensor(target_indices, dtype=torch.int16),
        "reference_indices": torch.tensor(reference_indices, dtype=torch.int16),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return tuple(payload["features"].shape)


def load_relation_shard(path, *, feature_dim=None):
    """Load and validate one tensor-only relation shard."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != RELATION_SHARD_KEYS:
        raise ValueError(f"Invalid relation shard fields in {path}")
    rows = payload["features"].shape[0]
    if payload["features"].ndim != 2 or not rows:
        raise ValueError(f"Invalid relation feature matrix in {path}")
    if feature_dim is not None and payload["features"].shape[1] != feature_dim:
        raise ValueError(f"Relation feature dimension mismatch in {path}")
    expected_shapes = {
        "labels": (rows, len(CANONICAL_RELATIONS)),
        "image_indices": (rows,),
        "target_indices": (rows,),
        "reference_indices": (rows,),
    }
    if any(tuple(payload[name].shape) != shape for name, shape in expected_shapes.items()):
        raise ValueError(f"Misaligned relation shard tensors in {path}")
    if not torch.isfinite(payload["features"]).all():
        raise ValueError(f"Non-finite relation features in {path}")
    return payload


def evaluate_relation_heads(heads, cache_dir, *, batch_size=8192, device="cpu"):
    """Return per-relation binary top-1 accuracy over a relation cache."""
    import torch

    manifest = relation_cache_manifest(cache_dir)
    correct = {relation: 0 for relation in CANONICAL_RELATIONS}
    positive = {relation: 0 for relation in CANONICAL_RELATIONS}
    predicted_positive = {relation: 0 for relation in CANONICAL_RELATIONS}
    true_positive = {relation: 0 for relation in CANONICAL_RELATIONS}
    count = 0
    heads.eval()
    with torch.inference_mode():
        for path in relation_shards(cache_dir):
            payload = load_relation_shard(path, feature_dim=manifest["feature_dim"])
            for start in range(0, len(payload["features"]), batch_size):
                features = payload["features"][start:start + batch_size].to(
                    device=device, dtype=torch.float32
                )
                labels = payload["labels"][start:start + batch_size].to(
                    device=device, dtype=torch.long
                )
                for column, relation in enumerate(CANONICAL_RELATIONS):
                    predictions = heads[relation](features).argmax(dim=1)
                    truth = labels[:, column]
                    correct[relation] += int((predictions == truth).sum())
                    positive[relation] += int(truth.sum())
                    predicted_positive[relation] += int(predictions.sum())
                    true_positive[relation] += int(((predictions == 1) & (truth == 1)).sum())
                count += len(features)
    metrics = {}
    for relation in CANONICAL_RELATIONS:
        precision_denominator = predicted_positive[relation]
        recall_denominator = positive[relation]
        precision = (
            true_positive[relation] / precision_denominator
            if precision_denominator else 0.0
        )
        recall = (
            true_positive[relation] / recall_denominator
            if recall_denominator else 0.0
        )
        metrics[relation] = correct[relation] / count
        metrics[f"{relation}_precision"] = precision
        metrics[f"{relation}_recall"] = recall
    metrics["macro"] = sum(metrics[relation] for relation in CANONICAL_RELATIONS) / len(CANONICAL_RELATIONS)
    metrics["pairs"] = count
    return metrics


def train_relation_heads(
    train_cache,
    val_cache,
    output,
    *,
    epochs=10,
    batch_size=8192,
    learning_rate=1e-3,
    weight_decay=1e-4,
    device="cpu",
    seed=0,
    progress=print,
):
    """Train binary right/front heads and save the best validation model."""
    import torch
    from torch.nn import functional as F

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Invalid relation-head training hyperparameters")
    train_manifest = relation_cache_manifest(train_cache)
    val_manifest = relation_cache_manifest(val_cache)
    comparable_keys = ("feature_dim", "object_feature_dim", "scene_feature_dim", "extractor")
    for key in comparable_keys:
        if train_manifest.get(key) != val_manifest.get(key):
            raise ValueError(f"Train and validation relation caches differ on {key}")

    torch.manual_seed(seed)
    heads = build_relation_heads(train_manifest["feature_dim"]).to(device)
    optimizer = torch.optim.AdamW(
        heads.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    shard_paths = relation_shards(train_cache)
    best_accuracy = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        heads.train()
        epoch_loss = 0.0
        pair_count = 0
        shuffled_paths = list(shard_paths)
        random.Random(seed + epoch).shuffle(shuffled_paths)
        for path in shuffled_paths:
            payload = load_relation_shard(path, feature_dim=train_manifest["feature_dim"])
            order = torch.randperm(len(payload["features"]), generator=generator)
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                features = payload["features"][indices].to(
                    device=device, dtype=torch.float32
                )
                labels = payload["labels"][indices].to(device=device, dtype=torch.long)
                loss = sum(
                    F.cross_entropy(heads[relation](features), labels[:, column])
                    for column, relation in enumerate(CANONICAL_RELATIONS)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(features)
                pair_count += len(features)

        metrics = evaluate_relation_heads(
            heads, val_cache, batch_size=batch_size, device=device
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = epoch_loss / pair_count
        history.append(metrics)
        progress(
            f"epoch {epoch}: loss={metrics['train_loss']:.4f}, "
            + ", ".join(
                f"{relation}={metrics[relation]:.2%}"
                for relation in (*CANONICAL_RELATIONS, "macro")
            )
        )
        if metrics["macro"] > best_accuracy:
            best_accuracy = metrics["macro"]
            best_state = deepcopy(heads.state_dict())
            checkpoint = {
                "schema_version": 1,
                "feature_dim": train_manifest["feature_dim"],
                "object_feature_dim": train_manifest["object_feature_dim"],
                "scene_feature_dim": train_manifest["scene_feature_dim"],
                "relations": list(CANONICAL_RELATIONS),
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


def load_relation_classifier(checkpoint_path, *, device="cpu"):
    """Load trained right/front heads as an ordered-pair classifier."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != 1
        or tuple(checkpoint.get("relations", ())) != CANONICAL_RELATIONS
    ):
        raise ValueError("Unsupported relation-head checkpoint")
    heads = build_relation_heads(checkpoint["feature_dim"])
    heads.load_state_dict(checkpoint["model_state_dict"])
    return TorchRelationClassifier(heads, device=device)
