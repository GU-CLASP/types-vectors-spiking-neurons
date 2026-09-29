"""Faster R-CNN adapter and the boundary between perception and categorical TTR.

The detector localizes entities. An injected attribute classifier supplies CLEVR
distributions for each crop; this is deliberately separate from COCO class IDs.
No weights or data are downloaded by this module.
"""

from dataclasses import dataclass
from math import ceil, floor, isfinite, isclose
from typing import Callable, Mapping

from .clevr import ATTRIBUTES, CLEVRModel


def _validate_box(box):
    if len(box) != 4:
        raise ValueError("Expected an xyxy box with four coordinates")
    x1, y1, x2, y2 = box
    if not all(isfinite(v) for v in box) or not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError("Expected a nonempty xyxy box with nonnegative coordinates")


def _validate_attribute_distributions(attributes):
    for attr, distribution in attributes.items():
        if attr not in ATTRIBUTES or not set(distribution) <= set(ATTRIBUTES[attr]):
            raise ValueError(f"Unknown CLEVR attribute labels: {attr}")
        if not distribution or any(not isfinite(p) or not 0 <= p <= 1
                                   for p in distribution.values()):
            raise ValueError("Attribute probabilities must be finite and in [0, 1]")
        if not isclose(sum(distribution.values()), 1.0, abs_tol=1e-5):
            raise ValueError("Each supplied attribute distribution must sum to one")


def scene_object_to_box(scene, obj, *, image_size=(480, 320)):
    """Estimate an ``xyxy`` box from a standard CLEVR scene annotation.

    The projection heuristic is adapted from Larry Chen's Apache-2.0-licensed
    ``extract_bounding_boxes`` implementation at the immutable source URL:
    https://github.com/larchen/clevr-vqa/blob/a224099addec82cf25f21d1fcbe11b15d3c02355/bounding_box.py

    Unlike that source, the rotation calculation retains both original world
    coordinates, output is scaled from CLEVR's 480x320 annotation coordinates
    to ``image_size``, and the result is clipped to the image boundary. These
    are approximate scene-derived regions, not mask-tight boxes.
    """
    if (not isinstance(image_size, tuple) or len(image_size) != 2
            or any(not isinstance(value, int) or value <= 0 for value in image_size)):
        raise ValueError("image_size must be a positive integer (width, height) tuple")
    try:
        pixel_x, pixel_y, _ = obj["pixel_coords"]
        world_x, world_y, world_z = obj["3d_coords"]
        right_x, right_y, _ = scene["directions"]["right"]
        shape = obj["shape"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("CLEVR box estimation needs pixel/3D coordinates, shape, and right direction") from exc
    values = (pixel_x, pixel_y, world_x, world_y, world_z, right_x, right_y)
    try:
        finite = all(isfinite(value) for value in values)
    except TypeError as exc:
        raise ValueError("CLEVR box inputs must be numeric") from exc
    if not finite:
        raise ValueError("CLEVR box inputs must be finite")
    if shape not in ATTRIBUTES["shape"]:
        raise ValueError(f"Unknown CLEVR shape: {shape}")

    rotated_y = -world_x * right_y + world_y * right_x
    half_height_down = 6.9 * world_z * (15 - rotated_y) / 2.0
    half_height_up = half_height_down
    half_width_left = half_height_down
    half_width_right = half_height_down

    if shape == "cylinder":
        distance = 9.4 + rotated_y
        object_height = 6.4
        if distance == 0 or 10 + rotated_y == 0:
            raise ValueError("Degenerate CLEVR cylinder projection")
        numerator = world_z * (object_height / distance + 1)
        denominator = numerator - world_z * (object_height - world_z) / distance
        if denominator == 0:
            raise ValueError("Degenerate CLEVR cylinder projection")
        half_height_up *= numerator / denominator
        half_height_down = half_height_up * (
            object_height - world_z + distance
        ) / (object_height + world_z + distance)
        half_width_left *= 11 / (10 + rotated_y)
        half_width_right = half_width_left
    elif shape == "cube":
        if 10 + rotated_y == 0:
            raise ValueError("Degenerate CLEVR cube projection")
        half_height_up *= 1.3 * 10 / (10 + rotated_y)
        half_height_down = half_height_up
        half_width_left = half_height_up
        half_width_right = half_height_up

    width, height = image_size
    scale_x = width / 480.0
    scale_y = height / 320.0
    box = (
        max(0.0, min(float(width), (pixel_x - half_width_left) * scale_x)),
        max(0.0, min(float(height), (pixel_y - half_height_down) * scale_y)),
        max(0.0, min(float(width), (pixel_x + half_width_right) * scale_x)),
        max(0.0, min(float(height), (pixel_y + half_height_up) * scale_y)),
    )
    _validate_box(box)
    return box


@dataclass(frozen=True)
class OracleRegion:
    """An externally supplied object identity and image region."""

    object_id: str
    box: tuple[float, float, float, float]

    def __post_init__(self):
        if not self.object_id:
            raise ValueError("Oracle region needs an object ID")
        _validate_box(self.box)


def clevr_scene_to_oracle_regions(scene, *, image_size=(480, 320)):
    """Create identity-aligned approximate regions for every object in a scene."""
    if "image_index" not in scene or "objects" not in scene:
        raise ValueError("CLEVR scene needs image_index and objects")
    return [
        OracleRegion(
            f'{scene["image_index"]}-{index}',
            scene_object_to_box(scene, obj, image_size=image_size),
        )
        for index, obj in enumerate(scene["objects"])
    ]


@dataclass(frozen=True)
class FeatureObject:
    """Individuated H-data backed by one cached real-valued feature vector."""

    object_id: str
    box: tuple[float, float, float, float]
    score: float
    features: tuple[float, ...]

    def __post_init__(self):
        if not self.object_id:
            raise ValueError("Feature object needs an object ID")
        _validate_box(self.box)
        if not isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Feature-object score must be finite and in [0, 1]")
        features = tuple(float(value) for value in self.features)
        if not features or any(not isfinite(value) for value in features):
            raise ValueError("Feature vector must be nonempty and finite")
        object.__setattr__(self, "features", features)


@dataclass(frozen=True)
class Detection:
    object_id: str
    box: tuple[float, float, float, float]
    score: float
    attributes: Mapping[str, Mapping[str, float]]

    def __post_init__(self):
        if not self.object_id:
            raise ValueError("Detection needs an object ID")
        _validate_box(self.box)
        if not isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Detection score must be finite and in [0, 1]")
        _validate_attribute_distributions(self.attributes)


class FasterRCNNFeatureExtractor:
    """Extract box-head embeddings for supplied oracle regions.

    This deliberately bypasses the region proposal and prediction stages. The
    caller owns the oracle boxes; the Faster R-CNN transform, backbone, RoI pool,
    and box head produce one cached vector per region.
    """

    def __init__(self, model, *, device="cpu"):
        for name in ("transform", "backbone", "roi_heads"):
            if not hasattr(model, name):
                raise TypeError(f"Faster R-CNN model is missing {name}")
        for name in ("box_roi_pool", "box_head"):
            if not hasattr(model.roi_heads, name):
                raise TypeError(f"Faster R-CNN roi_heads is missing {name}")
        self.model = model.to(device).eval()
        self.device = device

    def extract(self, image, regions):
        return self.extract_batch([image], [regions])[0]

    def extract_batch(self, images, region_batches):
        """Extract features for a batch while preserving scene/object order."""
        import torch

        images = list(images)
        region_batches = [list(regions) for regions in region_batches]
        if not images or len(images) != len(region_batches):
            raise ValueError("Expected one nonempty oracle-region batch per image")
        targets = []
        for image, regions in zip(images, region_batches, strict=True):
            if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
                raise ValueError("Expected a float RGB image tensor [3,H,W] in [0,1]")
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("Expected finite image values in [0,1]")
            if len({region.object_id for region in regions}) != len(regions):
                raise ValueError("Oracle region IDs must be unique within each image")
            height, width = image.shape[-2:]
            if any(region.box[2] > width or region.box[3] > height for region in regions):
                raise ValueError("Oracle boxes must lie within the image")
            targets.append({"boxes": torch.tensor(
                [region.box for region in regions], dtype=image.dtype, device=self.device
            ).reshape(-1, 4)})

        if not any(region_batches):
            return [[] for _ in images]
        images = [image.to(self.device) for image in images]
        with torch.inference_mode():
            image_list, targets = self.model.transform(images, targets)
            feature_maps = self.model.backbone(image_list.tensors)
            if torch.is_tensor(feature_maps):
                feature_maps = {"0": feature_maps}
            transformed_boxes = [target["boxes"] for target in targets]
            pooled = self.model.roi_heads.box_roi_pool(
                feature_maps, transformed_boxes, image_list.image_sizes
            )
            embeddings = self.model.roi_heads.box_head(pooled)
        region_count = sum(map(len, region_batches))
        if embeddings.ndim != 2 or embeddings.shape[0] != region_count:
            raise ValueError("Faster R-CNN box head must return one flat vector per region")

        result = []
        offset = 0
        for regions in region_batches:
            vectors = embeddings[offset:offset + len(regions)].detach().cpu()
            result.append([
                FeatureObject(
                    region.object_id,
                    region.box,
                    1.0,
                    tuple(float(value) for value in vector),
                )
                for region, vector in zip(regions, vectors, strict=True)
            ])
            offset += len(regions)
        return result


def build_factorized_attribute_heads(feature_dim):
    """Build one trainable linear head for each CLEVR attribute family."""
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError("feature_dim must be a positive integer")
    from torch import nn

    return nn.ModuleDict({
        attr: nn.Linear(feature_dim, len(values))
        for attr, values in ATTRIBUTES.items()
    })


class TorchAttributeClassifier:
    """Adapt trained factorized Torch heads to feature -> distributions."""

    def __init__(self, heads, *, device="cpu"):
        if set(heads) != set(ATTRIBUTES):
            raise ValueError("Attribute heads must cover all CLEVR attribute families")
        self.heads = heads.to(device).eval()
        self.device = device

    def __call__(self, features):
        import torch

        vector = torch.tensor(features, dtype=torch.float32, device=self.device)
        if vector.ndim != 1:
            raise ValueError("Expected one flat feature vector")
        result = {}
        with torch.inference_mode():
            for attr, values in ATTRIBUTES.items():
                logits = self.heads[attr](vector)
                if logits.shape != (len(values),):
                    raise ValueError(f"Head for {attr} returned the wrong number of logits")
                probabilities = logits.softmax(dim=-1).cpu()
                result[attr] = {
                    value: float(probability)
                    for value, probability in zip(values, probabilities, strict=True)
                }
        return result


class FasterRCNNDetector:
    """Wrap a torchvision detector and a crop -> attribute distributions callback.

    The callback receives a float RGB [C,H,W] crop in [0,1]. Missing attributes
    provide no positive evidence. A trained callback/checkpoint is future work.
    """

    def __init__(self, model, attribute_classifier: Callable, *, score_threshold=0.5, device="cpu"):
        if not 0 <= score_threshold <= 1:
            raise ValueError("Detection threshold must be in [0, 1]")
        self.model = model.to(device).eval()
        self.attribute_classifier = attribute_classifier
        self.score_threshold = score_threshold
        self.device = device

    def detect(self, image, *, image_id="image"):
        import torch

        image = image.to(self.device)
        if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
            raise ValueError("Expected a float RGB image tensor [3,H,W] in [0,1]")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Expected finite image values in [0,1]")
        with torch.inference_mode():
            prediction = self.model([image])[0]
            detections = []
            height, width = image.shape[-2:]
            for i, (box, score) in enumerate(zip(prediction["boxes"], prediction["scores"], strict=True)):
                score = float(score)
                if score < self.score_threshold:
                    continue
                x1, y1, x2, y2 = map(float, box)
                x1, x2 = max(0.0, x1), min(float(width), x2)
                y1, y2 = max(0.0, y1), min(float(height), y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[:, floor(y1):ceil(y2), floor(x1):ceil(x2)]
                attrs = self.attribute_classifier(crop)
                detections.append(Detection(f"{image_id}-{i}", (x1, y1, x2, y2), score, attrs))
        return detections


def build_untrained_faster_rcnn(*, num_classes=2, **kwargs):
    """Construct an offline architecture for plumbing tests, NOT meaningful predictions.

    Both weight arguments must be None: torchvision otherwise downloads backbone
    weights even when detector weights are disabled. Two classes = background/entity.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    return fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=num_classes, **kwargs
    )


def model_from_feature_objects(
    objects, attribute_classifier, *, attribute_threshold=0.5, detection_threshold=0.5
):
    """Build a categorical model whose classifiers consume cached vectors lazily."""
    if not callable(attribute_classifier):
        raise TypeError("attribute_classifier must be callable")
    if not 0 <= attribute_threshold <= 1 or not 0 <= detection_threshold <= 1:
        raise ValueError("Thresholds must be in [0, 1]")
    scene = {}
    for obj in objects:
        if obj.object_id in scene:
            raise ValueError(f"Duplicate feature-object ID: {obj.object_id}")
        if obj.score < detection_threshold:
            continue
        scene[obj.object_id] = {
            "id": obj.object_id,
            "box": obj.box,
            "score": obj.score,
            "features": obj.features,
        }

    distribution_cache = {}

    def distributions(obj):
        object_id = obj["id"]
        if object_id not in distribution_cache:
            predicted = attribute_classifier(obj["features"])
            _validate_attribute_distributions(predicted)
            distribution_cache[object_id] = {
                attr: dict(distribution) for attr, distribution in predicted.items()
            }
        return distribution_cache[object_id]

    def classified(evidence, attr, value):
        distribution = distributions(evidence[0]).get(attr, {})
        return value in distribution and distribution[value] > attribute_threshold

    classifiers = {
        value: (lambda evidence, a=attr, v=value: classified(evidence, a, v))
        for attr, values in ATTRIBUTES.items() for value in values
    }
    return CLEVRModel(scene, classifiers)


def model_from_detections(detections, *, attribute_threshold=0.5, detection_threshold=0.5):
    """Freeze a prediction snapshot and threshold scores into Boolean witness conditions.

    Spatial relations are intentionally absent. Image-plane box ordering is not a
    substitute for CLEVR's camera-relative 3D spatial relations.
    """
    if not 0 <= attribute_threshold <= 1 or not 0 <= detection_threshold <= 1:
        raise ValueError("Thresholds must be in [0, 1]")
    scene = {}
    seen = set()
    for detection in detections:
        if detection.object_id in seen:
            raise ValueError(f"Duplicate detection ID: {detection.object_id}")
        seen.add(detection.object_id)
        if detection.score < detection_threshold:
            continue
        scene[detection.object_id] = {
            "id": detection.object_id,
            "box": detection.box,
            "score": detection.score,
            "probabilities": {a: dict(ps) for a, ps in detection.attributes.items()},
        }

    def classified(evidence, attr, value):
        distribution = evidence[0]["probabilities"].get(attr, {})
        # A strict threshold avoids interpreting tied 0.5/0.5 predictions as two labels.
        return value in distribution and distribution[value] > attribute_threshold

    classifiers = {
        value: (lambda evidence, a=attr, v=value: classified(evidence, a, v))
        for attr, values in ATTRIBUTES.items() for value in values
    }
    return CLEVRModel(scene, classifiers)
