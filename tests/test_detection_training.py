import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from spinls.detection_training import (
    CLEVRDetectionDataset,
    DetectionMetrics,
    detect_entities,
)


class DetectionTrainingTests(unittest.TestCase):
    def test_dataset_builds_one_class_targets_from_scene_boxes(self):
        try:
            import torch
            from PIL import Image
        except ImportError:
            self.skipTest("Optional vision dependencies not installed")

        scene = {
            "image_index": 7,
            "image_filename": "CLEVR_train_000007.png",
            "directions": {"right": [1, 0, 0]},
            "objects": [{
                "pixel_coords": [240, 160, 8],
                "3d_coords": [1, 2, 0.35],
                "shape": "sphere",
            }],
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scenes").mkdir()
            (root / "images" / "train").mkdir(parents=True)
            (root / "scenes" / "CLEVR_train_scenes.json").write_text(
                json.dumps({"scenes": [scene]})
            )
            Image.new("RGB", (480, 320)).save(
                root / "images" / "train" / scene["image_filename"]
            )
            image, target = CLEVRDetectionDataset(root, "train")[0]

        self.assertEqual(tuple(image.shape), (3, 320, 480))
        self.assertEqual(target["image_id"].item(), 7)
        self.assertTrue(torch.equal(target["labels"], torch.tensor([1])))
        self.assertEqual(tuple(target["boxes"].shape), (1, 4))
        self.assertGreater(target["area"].item(), 0)

    def test_metrics_penalize_duplicate_and_missing_detections(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional torch dependency not installed")

        metrics = DetectionMetrics(iou_threshold=0.5, score_threshold=0.5)
        metrics.update(
            [
                {
                    "boxes": torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]]),
                    "scores": torch.tensor([0.9, 0.8]),
                    "labels": torch.tensor([1, 1]),
                },
                {
                    "boxes": torch.tensor([[0., 0., 10., 10.]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                },
            ],
            [
                {"boxes": torch.tensor([[0., 0., 10., 10.]])},
                {"boxes": torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]])},
            ],
        )
        result = metrics.compute()

        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertEqual(result["mean_absolute_count_error"], 1.0)

    def test_image_detections_receive_scene_local_ids(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional torch dependency not installed")

        class Detector(torch.nn.Module):
            def forward(self, images):
                return [{
                    "boxes": torch.tensor([[1., 2., 8., 9.], [0., 0., 2., 2.]]),
                    "scores": torch.tensor([0.9, 0.1]),
                    "labels": torch.tensor([1, 1]),
                }]

        detections = detect_entities(
            Detector(), torch.zeros(3, 10, 10), image_id="42", score_threshold=0.5
        )
        self.assertEqual([detection.object_id for detection in detections], ["42-d0"])
        self.assertEqual(detections[0].attributes, {})


if __name__ == "__main__":
    unittest.main()
