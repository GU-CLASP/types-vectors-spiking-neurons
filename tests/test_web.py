"""Smoke tests for the dependency-free CLEVR web demo."""

import json
from pathlib import Path
import tempfile
import unittest

from spinls.fixtures import tiny_question, tiny_scene
from spinls.web import CLEVRWebDemo


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
        (root / "questions" / "CLEVR_val_questions.json").write_text(
            json.dumps({"questions": [question, count_question]})
        )
        (root / "scenes" / "CLEVR_val_scenes.json").write_text(
            json.dumps({"scenes": [scene]})
        )
        (root / "images" / "val" / scene["image_filename"]).write_bytes(b"fake png")
        self.demo = CLEVRWebDemo(root)

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

    def test_answer_page_contains_take_and_answers(self):
        page = self.demo.render(scene_index=0, question_index=17, answer_requested=True)
        self.assertIn(r"\text{x}_{\text{1}} &amp;=&amp; \text{0-}_{\text{1}}", page)
        self.assertNotIn(r"\text{x1_x0}_{\text{right}} &amp;=&amp;", page)
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


if __name__ == "__main__":
    unittest.main()
