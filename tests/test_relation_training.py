import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from spinls.relation_training import (
    CANONICAL_RELATIONS,
    build_relation_heads,
    evaluate_relation_heads,
    load_relation_classifier,
    load_relation_shard,
    relation_cache_manifest,
    train_relation_heads,
    write_relation_shard,
)


class RelationTrainingTests(unittest.TestCase):
    def test_relation_shard_encodes_ordered_target_reference_pairs(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional torch dependency not installed")

        scene = {
            "image_index": 3,
            "objects": [{}, {}, {}],
            "relationships": {
                "right": [[1, 2], [], []],
                "front": [[2], [0], []],
            },
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shard-000003-000003.pt"
            shape = write_relation_shard(
                path,
                [(scene, [(1.0,), (2.0,), (3.0,)], (9.0, 10.0))],
            )
            payload = load_relation_shard(path, feature_dim=4)

        self.assertEqual(shape, (6, 4))
        self.assertEqual(payload["features"][0].tolist(), [2.0, 1.0, 9.0, 10.0])
        self.assertEqual(payload["target_indices"][0].item(), 1)
        self.assertEqual(payload["reference_indices"][0].item(), 0)
        self.assertEqual(payload["labels"][0].tolist(), [1, 0])
        self.assertEqual(payload["features"][1].tolist(), [3.0, 1.0, 9.0, 10.0])
        self.assertEqual(payload["labels"][1].tolist(), [1, 1])

    def test_relation_heads_train_and_load(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional torch dependency not installed")

        scenes = []
        for image_index in range(12):
            scene = {
                "image_index": image_index,
                "objects": [{}, {}, {}],
                "relationships": {
                    "right": [[1, 2], [2], []],
                    "front": [[2], [], []],
                },
            }
            object_features = [(1.0, 0.0), (0.0, 1.0), (0.0, 2.0)]
            scene_features = (float(image_index % 2),)
            scenes.append((scene, object_features, scene_features))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val"):
                cache = root / split
                cache.mkdir()
                (cache / "manifest.json").write_text(json.dumps({
                    "schema_version": 1,
                    "relations": list(CANONICAL_RELATIONS),
                    "object_feature_dim": 2,
                    "scene_feature_dim": 1,
                    "feature_dim": 5,
                    "extractor": {"name": "synthetic"},
                }))
                write_relation_shard(cache / "shard-000000-000011.pt", scenes)

            self.assertEqual(relation_cache_manifest(root / "train")["feature_dim"], 5)
            heads, _ = train_relation_heads(
                root / "train",
                root / "val",
                root / "relations.pt",
                epochs=20,
                batch_size=32,
                learning_rate=0.05,
                weight_decay=0,
                progress=lambda message: None,
            )
            metrics = evaluate_relation_heads(heads, root / "val")
            self.assertGreaterEqual(metrics["macro"], 0.95)
            classifier = load_relation_classifier(root / "relations.pt")
            distribution = classifier((0.0, 1.0), (1.0, 0.0), (0.0,))
            self.assertEqual(set(distribution), set(CANONICAL_RELATIONS))
            self.assertGreater(distribution["right"][True], distribution["right"][False])

        with self.assertRaisesRegex(ValueError, "input_dim"):
            build_relation_heads(0)


if __name__ == "__main__":
    unittest.main()
