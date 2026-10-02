"""Lazy image-to-H-data backend for the CLEVR web comparison demo."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from .clevr import ATTRIBUTES
from .detection_training import detect_entities, load_clevr_detector
from .perception import (
    FasterRCNNFeatureExtractor,
    FeatureObject,
    OracleRegion,
    model_from_feature_objects,
)
from .vision_training import load_attribute_classifier


class VisualBackendUnavailable(RuntimeError):
    """A required trained model or perceptual classifier is unavailable."""


@dataclass(frozen=True)
class VisualPrediction:
    object_id: str
    box: tuple[float, float, float, float]
    detection_score: float
    attributes: dict[str, dict[str, float]]


@dataclass(frozen=True)
class VisualObservation:
    model: object
    predictions: tuple[VisualPrediction, ...]


class VisualCLEVRBackend:
    """Detect objects and classify their attributes from a CLEVR image."""

    def __init__(
        self,
        detector_checkpoint,
        attribute_checkpoint,
        *,
        device="cpu",
        detection_threshold=0.5,
        cache_size=16,
    ):
        if not 0 <= detection_threshold <= 1:
            raise ValueError("detection_threshold must be in [0, 1]")
        self.detector_checkpoint = Path(detector_checkpoint)
        self.attribute_checkpoint = Path(attribute_checkpoint)
        self.device = device
        self.detection_threshold = detection_threshold
        self.cache_size = cache_size
        self._detector = None
        self._feature_extractor = None
        self._attribute_classifier = None
        self._cache = {}
        self._lock = Lock()

    def status(self):
        missing = [
            str(path)
            for path in (self.detector_checkpoint, self.attribute_checkpoint)
            if not path.is_file()
        ]
        return {
            "available": not missing,
            "missing": missing,
            "relations_available": False,
        }

    def checkpoint_metadata(self):
        import torch

        metadata = {}
        if self.detector_checkpoint.is_file():
            checkpoint = torch.load(
                self.detector_checkpoint, map_location="cpu", weights_only=True
            )
            metadata["detector"] = {
                key: checkpoint.get(key)
                for key in ("architecture", "epoch", "validation", "min_size", "max_size")
            }
        if self.attribute_checkpoint.is_file():
            checkpoint = torch.load(
                self.attribute_checkpoint, map_location="cpu", weights_only=True
            )
            metadata["attributes"] = {
                key: checkpoint.get(key)
                for key in ("extractor", "best_epoch", "best_validation")
            }
        return metadata

    def _load(self):
        status = self.status()
        if not status["available"]:
            raise VisualBackendUnavailable(
                "Visual models are not installed: " + ", ".join(status["missing"])
            )
        if self._detector is not None:
            return

        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights,
            fasterrcnn_resnet50_fpn,
        )

        try:
            attribute_checkpoint = torch.load(
                self.attribute_checkpoint, map_location="cpu", weights_only=True
            )
            extractor = attribute_checkpoint.get("extractor", {})
            if (extractor.get("architecture") != "resnet50-fpn"
                    or extractor.get("weights") != "COCO_V1"):
                raise ValueError(
                    "the attribute checkpoint does not use COCO ResNet50-FPN features"
                )
            feature_model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            )
            detector = load_clevr_detector(
                self.detector_checkpoint, device=self.device
            )
            feature_extractor = FasterRCNNFeatureExtractor(
                feature_model, device=self.device
            )
            attribute_classifier = load_attribute_classifier(
                self.attribute_checkpoint, device=self.device
            )
        except (ImportError, KeyError, RuntimeError, ValueError) as exc:
            raise VisualBackendUnavailable(f"Could not load visual models: {exc}") from exc
        self._detector = detector
        self._feature_extractor = feature_extractor
        self._attribute_classifier = attribute_classifier

    def observe(self, image_path, image_id):
        """Return a cached perceptual scene generated only from image pixels."""
        cache_key = (str(Path(image_path).resolve()), str(image_id))
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
            self._load()

            from PIL import Image
            from torchvision.transforms.functional import pil_to_tensor

            with Image.open(image_path) as source:
                image = pil_to_tensor(source.convert("RGB")).float().div_(255)
            detections = detect_entities(
                self._detector,
                image,
                image_id=str(image_id),
                score_threshold=self.detection_threshold,
                device=self.device,
            )
            regions = [OracleRegion(detection.object_id, detection.box) for detection in detections]
            extracted = self._feature_extractor.extract(image, regions)
            feature_objects = [
                FeatureObject(
                    feature.object_id,
                    feature.box,
                    detection.score,
                    feature.features,
                )
                for feature, detection in zip(extracted, detections, strict=True)
            ]
            distributions = {
                obj.object_id: self._attribute_classifier(obj.features)
                for obj in feature_objects
            }
            model = model_from_feature_objects(
                feature_objects,
                lambda features: self._attribute_classifier(features),
                classification_mode="argmax",
                detection_threshold=0.0,
            )
            observation = VisualObservation(
                model=model,
                predictions=tuple(
                    VisualPrediction(
                        obj.object_id,
                        obj.box,
                        obj.score,
                        distributions[obj.object_id],
                    )
                    for obj in feature_objects
                ),
            )
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = observation
            return observation


def categorical_predictions(prediction):
    """Return top label and confidence for every mutually exclusive family."""
    result = {}
    for name in ATTRIBUTES:
        distribution = prediction.attributes.get(name, {})
        if distribution:
            label = max(distribution, key=distribution.get)
            result[name] = (label, distribution[label])
    return result
