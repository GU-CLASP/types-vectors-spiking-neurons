"""CLEVR object-detection data, metrics, and checkpoint helpers."""

import json
from pathlib import Path

from .perception import Detection, FasterRCNNDetector, clevr_scene_to_oracle_regions


DETECTOR_SCHEMA_VERSION = 1


class CLEVRDetectionDataset:
    """Present scene-derived CLEVR boxes as one-class detection targets."""

    def __init__(self, clevr_dir, split, *, limit=None):
        if split not in ("train", "val"):
            raise ValueError("Detection training requires the train or val split")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        self.clevr_dir = Path(clevr_dir)
        self.split = split
        scene_path = self.clevr_dir / "scenes" / f"CLEVR_{split}_scenes.json"
        with scene_path.open() as source:
            scenes = json.load(source)["scenes"]
        self.scenes = scenes[:limit]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor

        scene = self.scenes[index]
        image_path = self.clevr_dir / "images" / self.split / scene["image_filename"]
        with Image.open(image_path) as source:
            image = pil_to_tensor(source.convert("RGB")).float().div_(255)
        regions = clevr_scene_to_oracle_regions(
            scene, image_size=(image.shape[2], image.shape[1])
        )
        boxes = torch.tensor([region.box for region in regions], dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(scene["image_index"], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image, target


def detection_collate(batch):
    """Keep variable-length image targets separate for torchvision detectors."""
    return tuple(zip(*batch, strict=True))


def build_clevr_detector(*, pretrained=True, min_size=320, max_size=480):
    """Build a Faster R-CNN ResNet50-FPN with background/entity outputs."""
    if min_size <= 0 or max_size < min_size:
        raise ValueError("Expected 0 < min_size <= max_size")
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
    )
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, 2)
    return model


class DetectionMetrics:
    """Accumulate greedy one-to-one detection metrics at a fixed IoU threshold."""

    def __init__(self, *, iou_threshold=0.5, score_threshold=0.5):
        if not 0 <= iou_threshold <= 1 or not 0 <= score_threshold <= 1:
            raise ValueError("Detection thresholds must be in [0, 1]")
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.matched_iou_sum = 0.0
        self.images = 0
        self.absolute_count_error = 0

    def update(self, predictions, targets):
        import torch
        from torchvision.ops import box_iou

        for prediction, target in zip(predictions, targets, strict=True):
            keep = (prediction["scores"] >= self.score_threshold) & (
                prediction["labels"] == 1
            )
            boxes = prediction["boxes"][keep]
            scores = prediction["scores"][keep]
            boxes = boxes[scores.argsort(descending=True)]
            truth = target["boxes"]
            unmatched = torch.ones(len(truth), dtype=torch.bool, device=truth.device)
            matched = 0
            for box in boxes:
                if not unmatched.any():
                    break
                overlaps = box_iou(box[None], truth)[0]
                overlaps[~unmatched] = -1
                overlap, truth_index = overlaps.max(dim=0)
                if float(overlap) >= self.iou_threshold:
                    unmatched[truth_index] = False
                    matched += 1
                    self.matched_iou_sum += float(overlap)
            self.true_positives += matched
            self.false_positives += len(boxes) - matched
            self.false_negatives += len(truth) - matched
            self.absolute_count_error += abs(len(boxes) - len(truth))
            self.images += 1

    def compute(self):
        precision_denominator = self.true_positives + self.false_positives
        recall_denominator = self.true_positives + self.false_negatives
        precision = self.true_positives / precision_denominator if precision_denominator else 0.0
        recall = self.true_positives / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": (
                self.matched_iou_sum / self.true_positives if self.true_positives else 0.0
            ),
            "mean_absolute_count_error": (
                self.absolute_count_error / self.images if self.images else 0.0
            ),
            "images": self.images,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


def evaluate_detector(
    model, data_loader, *, device="cpu", iou_threshold=0.5, score_threshold=0.5
):
    """Evaluate a detector without retaining all predictions in memory."""
    import torch

    metrics = DetectionMetrics(
        iou_threshold=iou_threshold, score_threshold=score_threshold
    )
    model.eval()
    with torch.inference_mode():
        for images, targets in data_loader:
            predictions = model([image.to(device) for image in images])
            cpu_predictions = [
                {name: value.detach().cpu() for name, value in prediction.items()}
                for prediction in predictions
            ]
            cpu_targets = [
                {name: value.detach().cpu() for name, value in target.items()}
                for target in targets
            ]
            metrics.update(cpu_predictions, cpu_targets)
    return metrics.compute()


def detector_checkpoint(model, *, epoch, metrics, min_size, max_size, optimizer=None,
                        scheduler=None, scaler=None):
    """Create a tensor-only checkpoint suitable for safe torch loading."""
    checkpoint = {
        "schema_version": DETECTOR_SCHEMA_VERSION,
        "architecture": "fasterrcnn_resnet50_fpn",
        "classes": ["background", "entity"],
        "box_source": "scene_object_to_box:v1",
        "min_size": min_size,
        "max_size": max_size,
        "epoch": epoch,
        "validation": dict(metrics),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    return checkpoint


def load_clevr_detector(checkpoint_path, *, device="cpu"):
    """Load a trained one-class CLEVR detector for image-only inference."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema_version") != DETECTOR_SCHEMA_VERSION
            or checkpoint.get("classes") != ["background", "entity"]):
        raise ValueError("Unsupported CLEVR detector checkpoint")
    model = build_clevr_detector(
        pretrained=False,
        min_size=checkpoint["min_size"],
        max_size=checkpoint["max_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def detect_entities(model, image, *, image_id="image", score_threshold=0.5,
                    device="cpu"):
    """Create scene-local individuals from image-only detector predictions."""
    detector = FasterRCNNDetector(
        model,
        attribute_classifier=lambda crop: {},
        score_threshold=score_threshold,
        device=device,
    )
    predictions = detector.detect(image, image_id=image_id)
    return [
        Detection(f"{image_id}-d{index}", prediction.box, prediction.score, {})
        for index, prediction in enumerate(predictions)
    ]
