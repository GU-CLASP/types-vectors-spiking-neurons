"""Faster R-CNN adapter and the boundary between perception and categorical TTR.

The detector localizes entities. An injected attribute classifier supplies CLEVR
distributions for each crop; this is deliberately separate from COCO class IDs.
No weights or data are downloaded by this module.
"""

from dataclasses import dataclass
from math import ceil, floor, isfinite, isclose
from typing import Callable, Mapping

from .clevr import ATTRIBUTES, CLEVRModel


@dataclass(frozen=True)
class Detection:
    object_id: str
    box: tuple[float, float, float, float]
    score: float
    attributes: Mapping[str, Mapping[str, float]]

    def __post_init__(self):
        if not self.object_id:
            raise ValueError("Detection needs an object ID")
        x1, y1, x2, y2 = self.box
        if not all(isfinite(v) for v in self.box) or not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("Expected a nonempty xyxy box with nonnegative coordinates")
        if not 0 <= self.score <= 1:
            raise ValueError("Detection score must be in [0, 1]")
        for attr, distribution in self.attributes.items():
            if attr not in ATTRIBUTES or not set(distribution) <= set(ATTRIBUTES[attr]):
                raise ValueError(f"Unknown CLEVR attribute labels: {attr}")
            if not distribution or any(not 0 <= p <= 1 for p in distribution.values()):
                raise ValueError("Attribute probabilities must be in [0, 1]")
            if not isclose(sum(distribution.values()), 1.0, abs_tol=1e-5):
                raise ValueError("Each supplied attribute distribution must sum to one")


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
