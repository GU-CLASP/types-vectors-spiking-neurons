"""Smoke tests for the dependency-free CLEVR web demo."""

import json
from pathlib import Path
import tempfile
import unittest

from spinls.fixtures import tiny_question, tiny_scene
from spinls.perception import FeatureObject, model_from_feature_objects
from spinls.visual_backend import VisualObservation, VisualPrediction
from spinls.web import CLEVRWebDemo


class FakeVisualBackend:
    def checkpoint_metadata(self):
        return {}

    def observe(self, image_path, image_id):
        objects = [
            FeatureObject("0-d0", (80, 110, 160, 190), 0.96, (1.0, 0.0)),
            FeatureObject("0-d1", (280, 110, 370, 200), 0.88, (0.0, 1.0)),
        ]

        def classify(features):
            first = features[0] > features[1]
            return {
                "color": {"red": 0.81 if first else 0.09, "blue": 0.19 if first else 0.91},
                "size": {"small": 0.73 if first else 0.21, "large": 0.27 if first else 0.79},
                "material": {"rubber": 0.68 if first else 0.14, "metal": 0.32 if first else 0.86},
                "shape": {"cube": 0.94 if first else 0.08, "sphere": 0.06 if first else 0.92},
            }

        predictions = tuple(
            VisualPrediction(obj.object_id, obj.box, obj.score, classify(obj.features))
            for obj in objects
        )
        return VisualObservation(
            model_from_feature_objects(objects, classify, classification_mode="argmax"),
            predictions,
        )


class WebDemoTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        (root / "questions").mkdir()
        (root / "scenes").mkdir()
        (root / "images" / "val").mkdir(parents=True)

        scene = tiny_scene()
        scene.update({"split": "val", "image_filename": "CLEVR_val_000000.png"})
        for index, obj in enumerate(scene["objects"]):
            obj["pixel_coords"] = [100 + index * 200, 150, 5]
        question = tiny_question()
        question.update({"question_index": 17, "image_index": 0, "answer": "blue"})
        count_question = {
            "question": "How many objects are there?",
            "question_index": 18,
            "image_index": 0,
            "answer": "2",
            "program": [
                {"function": "scene", "value_inputs": [], "inputs": []},
                {"function": "count", "value_inputs": [], "inputs": [0]},
            ],
        }
        attribute_question = {
            "question": "What color is the cube?",
            "question_index": 19,
            "image_index": 0,
            "answer": "red",
            "program": [
                {"function": "scene", "value_inputs": [], "inputs": []},
                {"function": "filter_shape", "value_inputs": ["cube"], "inputs": [0]},
                {"function": "unique", "value_inputs": [], "inputs": [1]},
                {"function": "query_color", "value_inputs": [], "inputs": [2]},
            ],
        }
        (root / "questions" / "CLEVR_val_questions.json").write_text(
            json.dumps({"questions": [question, count_question, attribute_question]})
        )
        (root / "scenes" / "CLEVR_val_scenes.json").write_text(
            json.dumps({"scenes": [scene]})
        )
        (root / "images" / "val" / scene["image_filename"]).write_bytes(b"fake png")
        self.demo = CLEVRWebDemo(root)
        self.visual_demo = CLEVRWebDemo(root, visual_backend=FakeVisualBackend())

    def tearDown(self):
        self.tempdir.cleanup()

    def test_page_contains_scene_controls_and_question_latex(self):
        page = self.demo.render(scene_index=0, question_index=17)
        self.assertIn("CLEVR scene 0", page)
        self.assertIn("Object IDs", page)
        self.assertIn('aria-label="CLEVR image index"', page)
        self.assertIn('name="scene" type="number"', page)
        self.assertIn("0-0", page)
        self.assertIn("What color is the sphere right of the red cube?", page)
        self.assertIn("Question meaning type", page)
        self.assertIn(r"T_{\mathrm{col}}", page)
        self.assertIn(r"\text{intrp}", page)
        self.assertIn("Generate answer", page)
        self.assertIn("Ground truth", page)
        self.assertIn("Model predictions", page)
        self.assertIn('href="/info"', page)

    def test_answer_page_contains_take_and_answers(self):
        page = self.demo.render(scene_index=0, question_index=17, answer_requested=True)
        self.assertIn(r"\text{x}_{\text{1}} &amp;=&amp; \text{0-}_{\text{1}}", page)
        self.assertIn(r"\text{x1_x0}_{\text{right}} &amp;=&amp;", page)
        self.assertIn(r"\#\mathrm{json}(\text{0-1})", page)
        self.assertIn("#json(id)</code> abbreviates", page)
        self.assertNotIn(r"\text{material} &amp;=&amp;", page)
        self.assertIn("Answer composition", page)
        self.assertIn(r"\text{blue}(\text{0-}_{\text{1}})", page)
        self.assertIn("<small>Answer</small>blue", page)
        self.assertIn("<small>Dataset</small>blue", page)
        self.assertEqual(self.demo.image_path(0).read_bytes(), b"fake png")

    def test_operational_questions_show_semantic_limitation(self):
        page = self.demo.render(scene_index=0, question_index=18, answer_requested=True)
        self.assertIn("paper-aligned intrp for count questions is not yet specified", page)
        self.assertNotIn("Answer composition", page)
        self.assertIn("<small>Answer</small>2", page)

    def test_visual_tab_shows_boxes_predictions_and_vector_evidence(self):
        page = self.visual_demo.render(
            scene_index=0, question_index=19, answer_requested=True, backend="visual"
        )
        self.assertIn('class="active" href="/?scene=0&amp;question=19&amp;backend=visual"', page)
        self.assertIn('class="detection-box"', page)
        self.assertIn("0-d0", page)
        self.assertIn("object 96%", page)
        self.assertIn("<b>red</b> 81%", page)
        self.assertIn(r"\#\mathrm{vector}(\text{0-d0})", page)
        self.assertIn("#vector(id)</code>", page)
        self.assertIn("<small>Answer</small>red", page)
        self.assertIn('name="backend" value="visual"', page)

    def test_visual_relation_question_reports_missing_relation_model(self):
        page = self.visual_demo.render(
            scene_index=0, question_index=17, answer_requested=True, backend="visual"
        )
        self.assertIn("learned spatial-relation classifier", page)
        self.assertIn("relation training is pending", page)
        self.assertIn("Detected objects and attributes", page)
        self.assertIn("Generate answer", page)
        self.assertIn(" disabled", page)
        self.assertNotIn("<small>Answer</small>blue", page)

    def test_info_page_documents_formalization_models_and_pending_metrics(self):
        page = self.visual_demo.render_info()
        self.assertIn("How the demo works", page)
        self.assertIn("TTR and H-data", page)
        self.assertIn("COCO-pretrained", page)
        self.assertIn("Faster R-CNN", page)
        self.assertIn("softmax heads", page)
        self.assertIn("pending", page)
        self.assertIn("src/spinls/clevr.py", page)
        self.assertIn("arxiv.org/abs/1506.01497", page)


if __name__ == "__main__":
    unittest.main()
