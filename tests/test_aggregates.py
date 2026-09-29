"""Count/existence over scene-relative selections, including empty results."""
import unittest

from pyttr.categorical import Fun, ListType

from spinls.clevr import AggregateLookup, AmbiguousReference, CLEVRModel, UnsupportedProgram, clevr_to_h_scene
from spinls.fixtures import tiny_scene


def question(*steps):
    return {"program": [
        {"function": name, "value_inputs": values, "inputs": [] if i == 0 else [i - 1]}
        for i, (name, values) in enumerate(steps)
    ]}


def program(steps):
    return {"program": [
        {"function": name, "value_inputs": values, "inputs": inputs}
        for name, inputs, values in steps
    ]}


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.model = CLEVRModel(clevr_to_h_scene(tiny_scene()))

    def test_empty_single_and_multiple_matches(self):
        for filters, count in (([], 2), ([("filter_color", ["red"])], 1),
                               ([("filter_color", ["green"])], 0)):
            for op, expected in (("count", str(count)), ("exist", "yes" if count else "no")):
                with self.subTest(op=op, count=count):
                    q = question(("scene", []), *filters, (op, []))
                    answer, takes = self.model.answer(q)
                    self.assertEqual(answer, expected)
                    self.assertIsInstance(takes, list)
                    self.assertEqual(len(takes), count)
                    rec = self.model.clevr_to_question_rec(q)
                    self.assertEqual(rec["clfr"].appc(takes).comps.pred.name, expected)
                    self.assertIn(op, rec["clfr"].to_latex(vars=[]))
                    self.assertNotIn("intrp", rec)
                    self.assertIn("not yet specified", rec["semantic_note"])

    def test_empty_scene(self):
        model = CLEVRModel({})
        for op, expected in (("count", "0"), ("exist", "no")):
            self.assertEqual(model.answer(question(("scene", []), (op, []))), (expected, []))

    def test_same_excludes_reference_and_relations_select_target(self):
        for operation, values, expected in (("same_shape", [], "0"), ("relate", ["right"], "1")):
            q = question(("scene", []), ("filter_color", ["red"]), ("unique", []),
                         (operation, values), ("count", []))
            answer, takes = self.model.answer(q)
            self.assertEqual(answer, expected)
            if takes:
                self.assertEqual(takes[0]["x1"], "0-1")

    def test_presupposition_failure_is_not_zero_or_no(self):
        for color in (None, "green"):
            filters = [] if color is None else [("filter_color", [color])]
            for op in ("count", "exist"):
                q = question(("scene", []), *filters, ("unique", []), ("same_shape", []), (op, []))
                with self.subTest(color=color, op=op), self.assertRaises(AmbiguousReference):
                    self.model.answer(q)

    def test_count_projects_distinct_target_ids(self):
        # Four records, but only two different objects in target field x.
        background = self.model.record_type({"x": self.model.ind, "y": self.model.ind})
        takes = list(self.model.build_witness_takes(background))
        self.assertEqual(len(takes), 4)
        clfr = Fun("rs", ListType(background), AggregateLookup(self.model, "count", "x"))
        self.assertEqual(clfr.appc(takes + takes).comps.pred.name, "2")

    def test_union_counts_set_union_and_accepts_later_filters(self):
        # Sphere and large select the same blue object; blue is counted once.
        q = program([
            ("scene", [], []),
            ("filter_shape", [0], ["sphere"]),
            ("scene", [], []),
            ("filter_size", [2], ["large"]),
            ("union", [1, 3], []),
            ("filter_color", [4], ["blue"]),
            ("count", [5], []),
        ])
        answer, takes = self.model.answer(q)
        self.assertEqual(answer, "1")
        self.assertEqual(len(takes), 1)
        self.assertEqual(takes[0]["x0"], "0-1")
        rec = self.model.clevr_to_question_rec(q)
        self.assertIn("v", rec["bg"].show())
        self.assertEqual(rec["clfr"].appc([type(takes[0])({"target": "0-1"})]).comps.pred.name, "1")

    def test_union_preserves_branch_definite_description_checks(self):
        q = program([
            ("scene", [], []),
            ("filter_color", [0], ["red"]),
            ("unique", [1], []),
            ("relate", [2], ["right"]),
            ("scene", [], []),
            ("filter_color", [4], ["blue"]),
            ("union", [3, 5], []),
            ("count", [6], []),
        ])
        self.assertEqual(self.model.answer(q)[0], "1")
        ambiguous = tiny_scene()
        ambiguous["objects"].append(dict(ambiguous["objects"][0]))
        for relation in ambiguous["relationships"].values():
            relation.append([])
        with self.assertRaises(AmbiguousReference):
            CLEVRModel(clevr_to_h_scene(ambiguous)).answer(q)

    def test_count_comparisons_keep_their_two_witness_sets_separate(self):
        # The left and right count inputs are separate program branches, not a
        # conjunction over one record.  This also exercises an empty selection.
        cases = (
            ("less_than", ["green"], [], "yes"),
            ("greater_than", [], ["red"], "yes"),
            ("equal_integer", ["red"], ["blue"], "yes"),
            ("equal_integer", ["red"], [], "no"),
        )
        for operation, left_filters, right_filters, expected in cases:
            with self.subTest(operation=operation, left=left_filters, right=right_filters):
                steps = [("scene", [], [])]
                left = 0
                for color in left_filters:
                    steps.append(("filter_color", [left], [color]))
                    left = len(steps) - 1
                steps.append(("count", [left], []))
                left_count = len(steps) - 1
                steps.append(("scene", [], []))
                right = len(steps) - 1
                for color in right_filters:
                    steps.append(("filter_color", [right], [color]))
                    right = len(steps) - 1
                steps.append(("count", [right], []))
                right_count = len(steps) - 1
                steps.append((operation, [left_count, right_count], []))
                q = program(steps)
                answer, takes = self.model.answer(q)
                self.assertEqual(answer, expected)
                self.assertIsInstance(takes, list)
                rec = self.model.clevr_to_question_rec(q)
                self.assertIn(operation, rec["clfr"].to_latex(vars=[]))
                self.assertIn("negative-answer semantics", rec["semantic_note"])

    def test_malformed_or_nonterminal_aggregates_rejected(self):
        cases = [
            question(("scene", []), ("count", ["extra"])),
            question(("scene", []), ("unique", []), ("exist", [])),
            question(("scene", []), ("count", []), ("exist", [])),
            question(("scene", []), ("unique", []), ("query_size", []), ("count", [])),
            program([("scene", [], []), ("scene", [], []), ("union", [0, 1], []),
                     ("unique", [2], [])]),
        ]
        for q in cases:
            with self.subTest(program=q), self.assertRaises(UnsupportedProgram):
                self.model.clevr_to_question_rec(q)


if __name__ == "__main__":
    unittest.main()
