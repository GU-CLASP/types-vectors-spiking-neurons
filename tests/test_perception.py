import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from spinls.perception import (
    Detection,
    FasterRCNNDetector,
    FasterRCNNFeatureExtractor,
    FeatureObject,
    OracleRegion,
    TorchAttributeClassifier,
    build_factorized_attribute_heads,
    clevr_scene_to_oracle_regions,
    model_from_detections,
    model_from_feature_objects,
    scene_object_to_box,
)
from spinls.vision_training import (
    evaluate_attribute_heads,
    load_attribute_classifier,
    load_feature_shard,
    train_attribute_heads,
    write_feature_shard,
)


class PerceptionTests(unittest.TestCase):
    def test_scene_annotations_produce_scaled_identity_aligned_boxes(self):
        obj = {
            "pixel_coords": [240, 160, 8],
            "3d_coords": [1, 2, 0.35],
            "shape": "sphere",
        }
        scene = {
            "image_index": 12,
            "objects": [obj],
            "directions": {"right": [0, 1, 0]},
        }

        box = scene_object_to_box(scene, obj)
        expected = (220.68, 140.68, 259.32, 179.32)
        for actual, expected_value in zip(box, expected, strict=True):
            self.assertAlmostEqual(actual, expected_value)
        scaled = scene_object_to_box(scene, obj, image_size=(960, 640))
        for actual, expected_value in zip(scaled, expected, strict=True):
            self.assertAlmostEqual(actual, expected_value * 2)
        regions = clevr_scene_to_oracle_regions(scene)
        self.assertEqual(regions, [OracleRegion("12-0", box)])

    def test_scene_boxes_are_clipped_and_validate_required_geometry(self):
        obj = {
            "pixel_coords": [2, 2, 8],
            "3d_coords": [0, 0, 0.7],
            "shape": "cube",
        }
        scene = {
            "image_index": 0,
            "objects": [obj],
            "directions": {"right": [1, 0, 0]},
        }

        box = scene_object_to_box(scene, obj)
        self.assertEqual(box[:2], (0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "pixel/3D coordinates"):
            scene_object_to_box(scene, {"shape": "cube"})
        with self.assertRaisesRegex(ValueError, "image_size"):
            scene_object_to_box(scene, obj, image_size=(0, 320))

    def test_vector_attributes_are_classified_lazily_and_cached(self):
        objects = [
            FeatureObject("a", (0, 0, 10, 10), 1.0, (1.0, 0.0)),
            FeatureObject("b", (10, 0, 20, 10), 1.0, (0.0, 1.0)),
        ]
        calls = []

        def attributes(features):
            calls.append(features)
            first = features[0] > features[1]
            return {
                "color": {"red": 0.9 if first else 0.1,
                          "blue": 0.1 if first else 0.9},
                "shape": {"cube": 0.9 if first else 0.1,
                          "sphere": 0.1 if first else 0.9},
            }

        model = model_from_feature_objects(objects, attributes, attribute_threshold=0.7)
        self.assertTrue(model.matches(color="red", shape="cube"))
        self.assertTrue(model.matches(color="blue", shape="sphere"))
        self.assertFalse(model.matches(color="red", shape="sphere"))
        self.assertEqual(calls, [(1.0, 0.0), (0.0, 1.0)])
        self.assertEqual(model.scene["a"]["features"], (1.0, 0.0))

    def test_categorical_feature_classifiers_use_one_argmax_label(self):
        objects = [FeatureObject("a", (0, 0, 10, 10), 1.0, (1.0,))]

        def attributes(features):
            return {"color": {"red": 0.40, "blue": 0.35, "green": 0.25}}

        model = model_from_feature_objects(
            objects, attributes, classification_mode="argmax"
        )
        self.assertTrue(model.matches(color="red"))
        self.assertFalse(model.matches(color="blue"))
        with self.assertRaisesRegex(ValueError, "classification_mode"):
            model_from_feature_objects(objects, attributes, classification_mode="binary")

    def test_factorized_torch_heads_supply_distributions(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional torch dependency not installed")

        heads = build_factorized_attribute_heads(2)
        with torch.no_grad():
            for head in heads.values():
                head.weight.zero_()
                head.bias.copy_(torch.arange(len(head.bias), dtype=head.bias.dtype))
        distributions = TorchAttributeClassifier(heads)((0.25, 0.75))

        self.assertEqual(set(distributions), {"color", "size", "material", "shape"})
        for distribution in distributions.values():
            self.assertAlmostEqual(sum(distribution.values()), 1.0)

    def test_oracle_regions_are_encoded_by_the_box_head(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional torch dependency not installed")

        class Transform:
            def __call__(self, images, targets=None):
                image_list = type("ImageList", (), {
                    "tensors": torch.stack(images),
                    "image_sizes": [tuple(images[0].shape[-2:])],
                })()
                return image_list, targets

        class Backbone(torch.nn.Module):
            def forward(self, images):
                return {"0": images}

        class Pool(torch.nn.Module):
            def forward(self, features, boxes, image_sizes):
                return torch.cat(boxes)[:, :, None, None]

        class BoxHead(torch.nn.Module):
            def forward(self, pooled):
                return pooled.flatten(1)

        class ROIHeads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.box_roi_pool = Pool()
                self.box_head = BoxHead()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transform = Transform()
                self.backbone = Backbone()
                self.roi_heads = ROIHeads()

        extractor = FasterRCNNFeatureExtractor(Model())
        regions = [
            OracleRegion("image-0", (1, 2, 8, 9)),
            OracleRegion("image-1", (10, 3, 18, 12)),
        ]
        objects = extractor.extract(torch.zeros(3, 20, 20), regions)
        scene_features = extractor.extract_scene_features(torch.ones(3, 20, 20))

        self.assertEqual([obj.object_id for obj in objects], ["image-0", "image-1"])
        self.assertEqual(objects[0].features, (1.0, 2.0, 8.0, 9.0))
        self.assertEqual(scene_features, (1.0, 1.0, 1.0))
        batches = extractor.extract_batch(
            [torch.zeros(3, 20, 20), torch.zeros(3, 20, 20)],
            [[regions[0]], [regions[1]]],
        )
        self.assertEqual(
            [[obj.object_id for obj in batch] for batch in batches],
            [["image-0"], ["image-1"]],
        )
        with self.assertRaisesRegex(ValueError, "within the image"):
            extractor.extract(
                torch.zeros(3, 20, 20), [OracleRegion("bad", (1, 2, 21, 9))]
            )

    def test_feature_cache_trains_and_loads_attribute_heads(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional torch dependency not installed")

        attributes = {
            "color": ["purple", "brown", "gray", "yellow", "green", "blue", "cyan", "red"],
            "size": ["large", "small"],
            "material": ["metal", "rubber"],
            "shape": ["sphere", "cube", "cylinder"],
        }
        values = list(attributes.values())
        offsets = (0, 8, 10, 12)
        scenes = []
        for image_index in range(24):
            labels = [image_index % len(options) for options in values]
            vector = [0.0] * 15
            for offset, label in zip(offsets, labels, strict=True):
                vector[offset + label] = 4.0
            obj = {
                name: options[label]
                for name, options, label in zip(attributes, values, labels, strict=True)
            }
            scene = {"image_index": image_index, "objects": [obj]}
            feature = FeatureObject(
                f"{image_index}-0", (0, 0, 10, 10), 1.0, tuple(vector)
            )
            scenes.append((scene, [feature]))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            for split, batches in (("train", scenes), ("val", scenes)):
                cache = root / split
                write_feature_shard(cache / "shard-000000-000023.pt", batches)
                (cache / "manifest.json").write_text(json.dumps({
                    "schema_version": 1,
                    "split": split,
                    "feature_dim": 15,
                    "attributes": attributes,
                    "extractor": {"name": "synthetic"},
                }))
            payload = load_feature_shard(root / "train" / "shard-000000-000023.pt")
            self.assertEqual(tuple(payload["features"].shape), (24, 15))

            checkpoint = root / "heads.pt"
            heads, _ = train_attribute_heads(
                root / "train", root / "val", checkpoint,
                epochs=20, batch_size=24, learning_rate=0.05,
                weight_decay=0, progress=lambda message: None,
            )
            metrics = evaluate_attribute_heads(heads, root / "val")
            self.assertEqual(metrics["macro"], 1.0)
            distributions = load_attribute_classifier(checkpoint)(scenes[0][1][0].features)
            self.assertEqual(max(distributions["color"], key=distributions["color"].get), "purple")

    def test_external_probabilities_and_missing_attributes(self):
        detections = [Detection("a", (0, 0, 10, 10), 0.9,
                                {"color": {"red": 0.8, "blue": 0.2}, "shape": {"cube": 1.0}})]
        model = model_from_detections(detections, attribute_threshold=0.7)
        self.assertTrue(model.matches(color="red", shape="cube"))
        self.assertFalse(model.matches(color="blue"))
        self.assertFalse(model.matches(material="rubber"))
        self.assertFalse(model_from_detections(detections, attribute_threshold=0.9).matches(color="red"))
        self.assertFalse(model_from_detections(detections, detection_threshold=0.95).matches(color="red"))
        self.assertFalse(model_from_detections([]).matches())

    def test_validation(self):
        for attrs in ({"color": {"red": 0.3}}, {"shape": {"cat": 1.0}}, {"color": {"red": float("nan")}}):
            with self.assertRaises(ValueError):
                Detection("a", (0, 0, 10, 10), 0.9, attrs)
        detection = Detection("a", (0, 0, 10, 10), 0.9, {})
        with self.assertRaises(ValueError):
            model_from_detections([detection, detection])
        with self.assertRaises(ValueError):
            model_from_detections([], attribute_threshold=1.1)
        with self.assertRaises(ValueError):
            FeatureObject("a", (0, 0, 10, 10), 1.0, ())
        with self.assertRaises(ValueError):
            OracleRegion("a", (0, 0, 0, 10))
        with self.assertRaises(ValueError):
            build_factorized_attribute_heads(0)

    def test_detector_adapter_with_synthetic_network_output(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional torch dependency not installed")

        class Detector(torch.nn.Module):
            def forward(self, images):
                return [{"boxes": torch.tensor([[1., 2., 8., 9.], [0., 0., 2., 2.]]),
                         "scores": torch.tensor([0.9, 0.1]), "labels": torch.tensor([1, 1])}]

        shapes = []

        def attributes(crop):
            shapes.append(tuple(crop.shape))
            return {"color": {"red": 1.0}}

        detector = FasterRCNNDetector(Detector(), attributes)
        detections = detector.detect(torch.zeros(3, 10, 10), image_id="test")
        self.assertEqual(shapes, [(3, 7, 7)])
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].object_id, "test-0")
        self.assertTrue(model_from_detections(detections).matches(color="red"))


if __name__ == "__main__":
    unittest.main()
