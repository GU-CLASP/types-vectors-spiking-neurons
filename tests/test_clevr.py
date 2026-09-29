import copy
import unittest

from pyttr.categorical import Rec

from spinls.clevr import AmbiguousReference, CLEVRModel, UnsupportedProgram, clevr_to_h_scene
from spinls.fixtures import tiny_question, tiny_scene
from examples.clevr_ground_truth import witness_assignment


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.model = CLEVRModel(clevr_to_h_scene(tiny_scene()))

    def test_answers_all_attributes_and_relation_direction(self):
        for attr, expected in {"color": "blue", "shape": "sphere", "size": "large", "material": "metal"}.items():
            with self.subTest(attr=attr):
                question = tiny_question(attr)
                answer, take = self.model.answer(question)
                self.assertEqual(answer, expected)
                self.assertIsInstance(take, Rec)
                self.assertIs(self.model.clevr_to_question_rec(question)["bg"].query(take), True)
                self.assertEqual(take["x1"], "0-1")
                self.assertEqual(witness_assignment(
                    self.model, self.model.clevr_to_question_rec(question), take
                ), "x0 = 0-0, x1 = 0-1")
        question = tiny_question()
        question["program"][4]["value_inputs"] = ["left"]
        with self.assertRaises(AmbiguousReference):
            self.model.answer(question)

    def test_conjunction_requires_same_object(self):
        self.assertTrue(self.model.matches(color="red", shape="cube"))
        self.assertFalse(self.model.matches(color="blue", shape="cube"))
        self.assertTrue(self.model.matches(color="blue", shape="sphere"))

    def test_same_attribute_relations(self):
        scene = {
            "image_index": 4,
            "objects": [
                {"color": "red", "shape": "cube", "size": "small", "material": "metal"},
                {"color": "green", "shape": "sphere", "size": "small", "material": "metal"},
                {"color": "yellow", "shape": "sphere", "size": "large", "material": "rubber"},
                {"color": "red", "shape": "cylinder", "size": "large", "material": "rubber"},
            ],
            "relationships": {relation: [[], [], [], []] for relation in
                              ("left", "right", "front", "behind")},
        }
        model = CLEVRModel(clevr_to_h_scene(scene))
        cases = (
            ("same_color", "red", "cube", None, "shape", "cylinder", "4-3"),
            ("same_shape", "green", "sphere", ("color", "yellow"), "size", "large", "4-2"),
            ("same_size", "red", "cube", None, "color", "green", "4-1"),
            ("same_material", "green", "sphere", None, "shape", "cube", "4-0"),
        )
        for same, color, shape, candidate_filter, query, expected, candidate_id in cases:
            with self.subTest(same=same):
                steps = [
                    ("scene", []), ("filter_color", [color]), ("filter_shape", [shape]),
                    ("unique", []), (same, []),
                ]
                if candidate_filter:
                    steps.append((f"filter_{candidate_filter[0]}", [candidate_filter[1]]))
                steps.extend((("unique", []), (f"query_{query}", [])))
                question = {"program": [
                    {"function": name, "value_inputs": values,
                     "inputs": [] if i == 0 else [i - 1]}
                    for i, (name, values) in enumerate(steps)
                ]}
                answer, take = model.answer(question)
                self.assertEqual(answer, expected)
                self.assertEqual(take["x1"], candidate_id)
                question_record = model.clevr_to_question_rec(question)
                self.assertIn(f"x1_x0_{same}", question_record["bg"].comps)
                self.assertIn("x1_x0_distinct", question_record["bg"].comps)

    def test_same_attribute_excludes_reference_object(self):
        steps = [
            ("scene", []), ("filter_color", ["red"]), ("filter_shape", ["cube"]),
            ("unique", []), ("same_shape", []), ("unique", []), ("query_color", []),
        ]
        question = {"program": [
            {"function": name, "value_inputs": values,
             "inputs": [] if i == 0 else [i - 1]}
            for i, (name, values) in enumerate(steps)
        ]}
        with self.assertRaises(AmbiguousReference):
            self.model.answer(question)

    def test_attribute_query_comparisons_compile_as_branches(self):
        scene = {
            "image_index": 7,
            "objects": [
                {"color": "gray", "shape": "cube", "size": "large", "material": "metal"},
                {"color": "cyan", "shape": "cube", "size": "small", "material": "metal"},
            ],
            "relationships": {relation: [[], []] for relation in
                              ("left", "right", "front", "behind")},
        }
        model = CLEVRModel(clevr_to_h_scene(scene))
        expected = {"color": "no", "shape": "yes", "size": "no", "material": "yes"}
        for attr, answer_expected in expected.items():
            with self.subTest(attr=attr):
                program = [
                    {"function": "scene", "inputs": [], "value_inputs": []},
                    {"function": "filter_color", "inputs": [0], "value_inputs": ["gray"]},
                    {"function": "unique", "inputs": [1], "value_inputs": []},
                    {"function": f"query_{attr}", "inputs": [2], "value_inputs": []},
                    {"function": "scene", "inputs": [], "value_inputs": []},
                    {"function": "filter_color", "inputs": [4], "value_inputs": ["cyan"]},
                    {"function": "unique", "inputs": [5], "value_inputs": []},
                    {"function": f"query_{attr}", "inputs": [6], "value_inputs": []},
                    {"function": f"equal_{attr}", "inputs": [3, 7], "value_inputs": []},
                ]
                answer, take = model.answer({"program": program})
                self.assertEqual(answer, answer_expected)
                self.assertEqual((take["x0"], take["x1"]), ("7-0", "7-1"))

    def test_partial_constraints_and_no_binding_mutation(self):
        typ = self.model.description_type(color="red")
        bindings = {"x": "0-1"}
        self.assertEqual(list(self.model.build_witness_takes(typ, bindings)), [])
        self.assertEqual(bindings, {"x": "0-1"})
        restricted = self.model.get_restricted_type(["x"], typ)
        self.assertIn("x_color", restricted.comps)
        first = list(self.model.build_witness_takes(typ))
        second = list(self.model.build_witness_takes(typ))
        self.assertEqual([r["x"] for r in first], ["0-0"])
        self.assertEqual([r["x"] for r in second], ["0-0"])

    def test_evidence_must_match_predicate_argument(self):
        # red(blue-object) must not accept evidence about a different red object.
        wrong = self.model.ptype("red", ["0-1"])
        self.assertIs(wrong.query((self.model.scene["0-0"],)), False)
        fake = dict(self.model.scene["0-1"], color="red")
        self.assertIs(wrong.query((fake,)), False)
        self.assertIs(wrong.query(None), False)

    def test_models_do_not_share_cached_judgements(self):
        self.assertTrue(self.model.matches(color="red", shape="cube"))
        changed = tiny_scene()
        changed["objects"][0]["color"] = "green"
        other = CLEVRModel(clevr_to_h_scene(changed))
        self.assertFalse(other.matches(color="red", shape="cube"))
        self.assertTrue(other.matches(color="green", shape="cube"))
        self.assertTrue(self.model.matches(color="red", shape="cube"))

    def test_uniqueness_is_checked_before_later_filters(self):
        scene = tiny_scene()
        scene["objects"].append(copy.deepcopy(scene["objects"][0]))
        # A second red cube has no right neighbour. Later constraints could hide
        # the ambiguity if we checked only the final set of witnesses.
        for relation in scene["relationships"].values():
            relation.append([])
        model = CLEVRModel(clevr_to_h_scene(scene))
        with self.assertRaises(AmbiguousReference):
            model.answer(tiny_question())

    def test_invalid_and_unsupported_programs_fail_explicitly(self):
        for operation in ("count", "intersect", "same_nonsense"):
            question = tiny_question()
            question["program"][-1]["function"] = operation
            with self.assertRaises(UnsupportedProgram):
                self.model.answer(question)
        question = tiny_question()
        question["program"][2]["inputs"] = [99]
        with self.assertRaises(UnsupportedProgram):
            self.model.answer(question)
        with self.assertRaises(UnsupportedProgram):
            self.model.answer({"program": []})

    def test_function_application_and_display(self):
        rec = self.model.clevr_to_question_rec(tiny_question())
        take = next(self.model.build_witness_takes(rec["bg"]))
        meaning = Rec({label: rec[label] for label in ("bg", "intrp", "clfr")})
        self.assertIs(rec["meaning_type"].query(meaning), True)
        predicate = rec["clfr"].appc(take)
        self.assertEqual(predicate.body.comps.pred.name, "blue")
        result = rec["intrp"].appc(take).appc(predicate)
        self.assertEqual(result.comps.pred.name, "blue")
        self.assertEqual(result.comps.args, ["0-1"])
        self.assertIsNone(rec["clfr"].appc(Rec({})))
        self.assertIn("T_{\\mathrm{col}}", rec["clfr"].to_latex(vars=[]))
        self.assertIn("P(r.x1)", rec["intrp"].to_latex(vars=[]))


if __name__ == "__main__":
    unittest.main()
