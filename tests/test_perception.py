import unittest

from spinls.perception import Detection, FasterRCNNDetector, model_from_detections


class PerceptionTests(unittest.TestCase):
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
