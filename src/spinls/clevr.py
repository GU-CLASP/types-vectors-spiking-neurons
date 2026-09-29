"""Categorical TTR grounding for a deliberately small CLEVR program fragment.

Evidence is a tuple of scene objects, checked against the predicate's arguments
as well as its external classifier. Create a new model when scene evidence or
classifiers change: PyTTR caches successful judgements.
"""

from copy import deepcopy
from dataclasses import dataclass
from itertools import count
from pathlib import Path
import json

from pyttr.categorical import (
    BType, Fun, FunType, JoinType, ListType, PType, Possibility, Pred,
    Rec, RecTy, RecType, Ty,
)

ATTRIBUTES = {
    "color": ("purple", "brown", "gray", "yellow", "green", "blue", "cyan", "red"),
    "size": ("large", "small"),
    "material": ("metal", "rubber"),
    "shape": ("sphere", "cube", "cylinder"),
}
RELATIONS = ("right", "behind", "front", "left")
SAME_RELATIONS = tuple(f"same_{attr}" for attr in ATTRIBUTES)
PREDICATE_ARITIES = (
    {value: 1 for values in ATTRIBUTES.values() for value in values}
    | {relation: 2 for relation in RELATIONS}
    | {relation: 2 for relation in SAME_RELATIONS}
    | {"distinct": 2}
    | {answer: 0 for answer in ("yes", "no")}
)


class UnsupportedProgram(ValueError):
    """The question is outside the supported program fragment."""


class AmbiguousReference(ValueError):
    """A CLEVR unique operation did not identify exactly one object."""


class CLEVR:
    """Read existing CLEVR files without downloading data."""

    def __init__(self, clevr_dir, split="val"):
        self.clevr_dir = Path(clevr_dir)
        self.split = split
        with (self.clevr_dir / "questions" / f"CLEVR_{split}_questions.json").open() as f:
            self.questions = json.load(f)["questions"]
        with (self.clevr_dir / "scenes" / f"CLEVR_{split}_scenes.json").open() as f:
            scenes = json.load(f)["scenes"]
        self.scenes = {scene["image_index"]: scene for scene in scenes}

    def get_image_path(self, record):
        return self.clevr_dir / "images" / record.get("split", self.split) / record["image_filename"]


def clevr_to_h_scene(scene):
    """Keep CLEVR's relation direction: relationships[r][i] lists r-of-i objects."""
    result = {}
    for i, obj in enumerate(scene["objects"]):
        obj = deepcopy(obj)
        obj["id"] = f'{scene["image_index"]}-{i}'
        for relation in RELATIONS:
            obj[relation] = [f'{scene["image_index"]}-{j}'
                             for j in scene["relationships"][relation][i]]
        result[obj["id"]] = obj
    return result


class PredicateApplication:
    """A substitutable application of an elliptical answer predicate."""

    def __init__(self, variable, referent):
        self.variable, self.referent = variable, referent

    def subst(self, variable, predicate):
        if variable != self.variable:
            return self
        if hasattr(predicate, "app"):
            return predicate.app(self.referent)
        return Ty

    def show(self):
        return f"{self.variable}({self.referent})"

    def to_latex(self, vars):
        return f"{self.variable}({self.referent})"


class AttributeInterpretation:
    """Construct the PType contributed by an elliptical attribute answer."""

    def __init__(self, label, answer_type, variable="r"):
        self.label = label
        self.answer_type = answer_type
        self.variable = variable

    def subst(self, variable, record):
        if variable != self.variable:
            return self
        return Fun(
            "P", self.answer_type,
            PredicateApplication("P", record[self.label]),
        )

    def show(self):
        return f"lambda P:{self.answer_type.show()} . P({self.variable}.{self.label})"

    def to_latex(self, vars):
        return (
            rf"\lambda P:{self.answer_type.to_latex(vars)}\ .\ "
            rf"P({self.variable}.{self.label})"
        )


class AttributeClassifier:
    """Apply T_attr to a referent and return its elliptical predicate meaning."""

    def __init__(self, model, attr, label, variable="r"):
        self.model, self.attr, self.label, self.variable = model, attr, label, variable

    def subst(self, variable, record):
        if variable != self.variable:
            return self
        obj_id = record[self.label]
        if obj_id not in self.model.scene:
            return Fun("x", self.model.ind, Ty)
        obj = self.model.scene[obj_id]
        value = self.model.attribute_value(obj, self.attr)
        return Fun("x", self.model.ind, self.model.ptype(value, ["x"]))

    def show(self):
        return f"T_{self.attr}({self.variable}.{self.label})"

    def to_latex(self, vars):
        abbreviation = {"color": "col", "material": "mat"}.get(self.attr, self.attr)
        return rf"T_{{\mathrm{{{abbreviation}}}}}({self.variable}.{self.label})"


class FunctionTypeFromBackground:
    """A dependent field body resolving T_bg to a concrete function type."""

    def __init__(self, range_type, variable="T_bg"):
        self.range_type, self.variable = range_type, variable

    def subst(self, variable, background):
        if variable == self.variable:
            return FunType(background, self.range_type)
        return self

    def show(self):
        return f"({self.variable}->{self.range_type.show()})"

    def to_latex(self, vars):
        return rf"({self.variable}\rightarrow {self.range_type.to_latex(vars)})"


class CompareLookup:
    """A Python-backed function body comparing queried attributes in a record."""

    def __init__(self, model, attr, left_label, right_label, variable="r"):
        self.model = model
        self.attr = attr
        self.left_label = left_label
        self.right_label = right_label
        self.variable = variable

    def subst(self, variable, record):
        if variable != self.variable:
            return self
        left = self.model.scene[record[self.left_label]]
        right = self.model.scene[record[self.right_label]]
        answer = "yes" if self.model.classifiers[f"same_{self.attr}"]((left, right)) else "no"
        return self.model.ptype(answer, [])

    def show(self):
        return f"compare_{self.attr}({self.variable}.{self.left_label}, {self.variable}.{self.right_label})"

    def to_latex(self, vars):
        return (rf"\operatorname{{compare}}_{{{self.attr}}}"
                rf"({self.variable}.{self.left_label},{self.variable}.{self.right_label})")


class AggregateLookup:
    """Summarize the complete witness list supplied by the answer workflow.

    The ListType domain checks individual records, not exhaustiveness. As with
    uniqueness, completeness relative to the scene remains a Python obligation.
    Results are answer-label types, not an object-language proof of cardinality
    or absence. In particular, perceptual `no` means no positive evidence found.
    """

    def __init__(self, model, operation, label, variable="rs"):
        self.model, self.operation, self.label, self.variable = model, operation, label, variable

    def subst(self, variable, records):
        if variable != self.variable:
            return self
        referents = {record[self.label] for record in records}
        if self.operation == "exist":
            return self.model.ptype("yes" if referents else "no", [])
        # Counts have no fixed upper bound; do not add a finite numeral inventory.
        return PType(Pred(str(len(referents)), []), [], poss=self.model.poss)

    def show(self):
        return f"{self.operation}({self.variable}, {self.label})"

    def to_latex(self, vars):
        return rf"\operatorname{{{self.operation}}}({self.variable}, {self.label})"


class CountComparisonLookup:
    """Compare the distinct target individuals from two complete witness lists."""

    def __init__(self, model, operation, left_labels, right_labels, variable="rs"):
        self.model = model
        self.operation = operation
        self.left_labels = tuple(left_labels)
        self.right_labels = tuple(right_labels)
        self.variable = variable

    @staticmethod
    def _count(records, labels):
        return len({record[label] for record in records for label in labels if label in record})

    def subst(self, variable, records):
        if variable != self.variable:
            return self
        left = self._count(records, self.left_labels)
        right = self._count(records, self.right_labels)
        comparisons = {
            "less_than": left < right,
            "greater_than": left > right,
            "equal_integer": left == right,
        }
        return self.model.ptype("yes" if comparisons[self.operation] else "no", [])

    def show(self):
        return f"{self.operation}({self.variable}, {','.join(self.left_labels)}, {','.join(self.right_labels)})"

    def to_latex(self, vars):
        return (rf"\operatorname{{{self.operation}}}"
                rf"({self.variable};{','.join(self.left_labels)};{','.join(self.right_labels)})")


@dataclass(frozen=True)
class _Selection:
    fields: dict
    label: str
    unique: bool = False
    unique_checks: tuple = ()


@dataclass(frozen=True)
class _UnionSelection:
    """A disjunction of set-valued selection branches."""

    branches: tuple


@dataclass(frozen=True)
class _AttributeQuery:
    selection: _Selection
    attr: str


@dataclass(frozen=True)
class _CountQuery:
    """A set selection awaiting a terminal integer comparison."""

    selection: _Selection | _UnionSelection


class CLEVRModel:
    """One scene snapshot, isolated PyTTR model, and external Boolean classifiers.

    Classifiers map predicate names to callables on tuples of object dictionaries.
    They can consult metadata or threshold a perceptual model's scores. These are
    categorical decisions, not probabilistic TTR judgements.
    """

    def __init__(self, scene, classifiers=None):
        self.scene = deepcopy(scene)
        if any(key != obj.get("id") for key, obj in self.scene.items()):
            raise ValueError("Scene keys must equal object IDs")
        self.poss = Possibility()
        self.ind = BType("Ind", poss=self.poss)
        for obj_id in self.scene:
            self.ind.judge(obj_id)
        self.classifiers = {
            value: (lambda evidence, a=attr, v=value: evidence[0].get(a) == v)
            for attr, values in ATTRIBUTES.items() for value in values
        }
        self.classifiers.update({
            rel: (lambda evidence, r=rel: evidence[0]["id"] in evidence[1].get(r, ()))
            for rel in RELATIONS
        })
        self.classifiers.update({
            f"same_{attr}": (
                lambda evidence, a=attr: any(
                    self.classifiers[value]((evidence[0],))
                    and self.classifiers[value]((evidence[1],))
                    for value in ATTRIBUTES[a]
                )
            )
            for attr in ATTRIBUTES
        })
        self.classifiers["distinct"] = lambda evidence: evidence[0]["id"] != evidence[1]["id"]
        self.classifiers.update({answer: lambda evidence: True for answer in ("yes", "no")})
        self.classifiers.update(classifiers or {})
        self.preds = {}
        for name in self.classifiers:
            if name not in PREDICATE_ARITIES:
                raise ValueError(f"No predicate arity registered for classifier: {name}")
            arity = PREDICATE_ARITIES[name]
            pred = Pred(name, [self.ind] * arity)
            pred.learn_witness_fun(lambda args, n=name: self._witness_type(n, args))
            self.preds[name] = pred

    def _witness_type(self, name, args):
        witness_type = BType(f"evidence:{name}({','.join(map(str, args))})", poss=self.poss)
        if not witness_type.witness_conditions:
            expected = tuple(args)

            def accepts(evidence):
                if not isinstance(evidence, tuple) or len(evidence) != len(expected):
                    return False
                if not all(isinstance(obj, dict) and obj.get("id") == obj_id
                           and obj == self.scene.get(obj_id)
                           for obj, obj_id in zip(evidence, expected)):
                    return False
                return bool(self.classifiers[name](evidence))

            witness_type.learn_witness_condition(accepts)
        return witness_type

    def ptype(self, name, args):
        return PType(self.preds[name], list(args), poss=self.poss)

    def attribute_value(self, obj, attr):
        values = [value for value in ATTRIBUTES[attr]
                  if self.classifiers[value]((obj,))]
        if len(values) != 1:
            raise AmbiguousReference(f"Expected one {attr} classification; found {values}")
        return values[0]

    def record_type(self, fields):
        return RecType(fields).in_poss(self.poss)

    def individual_labels(self, record_type):
        return [label for label, typ in record_type.comps.items() if typ is self.ind]

    def get_restricted_type(self, labels, record_type):
        """Retain assigned individuals and all constraints whose arguments are bound."""
        labels = set(labels)
        fields = {}
        for label, typ in record_type.comps.items():
            if typ is self.ind and label in labels:
                fields[label] = typ
            elif isinstance(typ, tuple) and all(arg in labels for arg in typ[1]):
                fields[label] = typ
            elif typ is not self.ind and not isinstance(typ, tuple):
                raise ValueError("Witness search supports flat individual/dependent fields only")
        return self.record_type(fields)

    def h_data_to_sit_take(self, label_map):
        fields = dict(label_map)
        for label, obj_id in label_map.items():
            obj = self.scene[obj_id]
            for attr in ATTRIBUTES:
                fields[f"{label}_{attr}"] = (obj,)
            for other_label, other_id in label_map.items():
                for relation in RELATIONS + SAME_RELATIONS + ("distinct",):
                    fields[f"{label}_{other_label}_{relation}"] = (obj, self.scene[other_id])
        return Rec(fields)

    def build_witness_takes(self, record_type, label_map=None):
        """Deterministic search with partial checks; never modify the caller's bindings.

        Distinct variables may denote the same individual, as in TTR. Irreflexive
        spatial predicates, rather than an implicit all-different rule, exclude it.
        """
        bindings = dict(label_map or {})
        labels = self.individual_labels(record_type)
        if not set(bindings) <= set(labels) or not set(bindings.values()) <= self.scene.keys():
            raise ValueError("Bindings must map individual fields to known scene IDs")

        def search(bound):
            take = self.h_data_to_sit_take(bound)
            restricted = self.get_restricted_type(bound, record_type)
            if restricted.query(take) is not True:
                return
            free = [label for label in labels if label not in bound]
            if not free:
                if record_type.query(take) is True:
                    yield take
                return
            for obj_id in self.scene:
                yield from search(bound | {free[0]: obj_id})

        yield from search(bindings)

    def _question_record(self, selection, body, semantic_note=None):
        background = self.record_type(selection.fields)
        fields = {"bg": background, "clfr": Fun("r", background, body),
                  "unique_checks": list(selection.unique_checks)}
        if semantic_note:
            fields["semantic_note"] = semantic_note
        return Rec(fields)

    def _attribute_question_record(self, query):
        background = self.record_type(query.selection.fields)
        answer_type = FunType(self.ind, Ty)
        intrp = Fun(
            "r", background,
            AttributeInterpretation(query.selection.label, answer_type),
        )
        clfr = Fun(
            "r", background,
            AttributeClassifier(self, query.attr, query.selection.label),
        )
        meaning = Rec({"bg": background, "intrp": intrp, "clfr": clfr})
        meaning_type = self.record_type({
            "bg": RecTy,
            "intrp": (
                Fun(
                    "T_bg", RecTy,
                    FunctionTypeFromBackground(FunType(answer_type, Ty)),
                ),
                ["bg"],
            ),
            "clfr": (
                Fun("T_bg", RecTy, FunctionTypeFromBackground(answer_type)),
                ["bg"],
            ),
        })
        return Rec({
            **meaning.as_dict(),
            "meaning_type": meaning_type,
            "unique_checks": list(query.selection.unique_checks),
        })

    def _aggregate_question_record(self, selection, operation):
        """Build a count/existence question record for one selection or a union."""
        if isinstance(selection, _Selection):
            background = self.record_type(selection.fields)
            return Rec({
                "bg": background,
                "clfr": Fun("rs", ListType(background).in_poss(self.poss),
                            AggregateLookup(self, operation, selection.label)),
                "unique_checks": list(selection.unique_checks),
                "aggregate": operation,
                "target_label": selection.label,
                "semantic_note": (
                    f"The paper-aligned intrp for {operation} questions is not yet specified; "
                    "this is an operational answer over the complete witness list."
                ),
            })

        backgrounds = [self.record_type(branch.fields) for branch in selection.branches]
        background = backgrounds[0]
        for branch_background in backgrounds[1:]:
            background = JoinType(background, branch_background).in_poss(self.poss)
        target_background = self.record_type({"target": self.ind})
        return Rec({
            "bg": background,
            "clfr": Fun("rs", ListType(target_background).in_poss(self.poss),
                        AggregateLookup(self, operation, "target")),
            "unique_checks": [check for branch in selection.branches
                              for check in branch.unique_checks],
            "aggregate": operation,
            "semantic_note": (
                f"The paper-aligned intrp for {operation} questions is not yet specified; "
                "this is an operational answer over the complete witness list."
            ),
            "branches": [
                {"background": branch_background, "label": branch.label}
                for branch_background, branch in zip(backgrounds, selection.branches)
            ],
        })

    def _count_comparison_record(self, operation, left, right):
        """Build a two-branch count comparison without conjoining its contexts."""
        branches = []
        for side, query in (("left", left), ("right", right)):
            selection = query.selection
            selections = (selection,) if isinstance(selection, _Selection) else selection.branches
            for selection_branch in selections:
                branches.append({
                    "side": side,
                    "background": self.record_type(selection_branch.fields),
                    "label": selection_branch.label,
                    "unique_checks": selection_branch.unique_checks,
                })
        background = branches[0]["background"]
        for branch in branches[1:]:
            background = JoinType(background, branch["background"]).in_poss(self.poss)
        return Rec({
            "bg": background,
            "clfr": Fun(
                "rs", ListType(background).in_poss(self.poss),
                CountComparisonLookup(
                    self, operation,
                    [branch["label"] for branch in branches if branch["side"] == "left"],
                    [branch["label"] for branch in branches if branch["side"] == "right"],
                ),
            ),
            "unique_checks": [check for branch in branches for check in branch["unique_checks"]],
            "comparison": operation,
            "semantic_note": (
                "The paper-aligned intrp and negative-answer semantics for integer "
                "comparisons are not yet specified; this is an operational answer."
            ),
            "branches": [{key: value for key, value in branch.items() if key != "unique_checks"}
                         for branch in branches],
        })

    @staticmethod
    def _set_selection(selection):
        """A union consumes selections as sets, retaining their presuppositions."""
        if isinstance(selection, _UnionSelection):
            return selection.branches
        if isinstance(selection, _Selection):
            return (_Selection(selection.fields, selection.label, False, selection.unique_checks),)
        raise UnsupportedProgram("union requires selection inputs")

    def _filter_selection(self, selection, attr, value):
        if attr not in ATTRIBUTES or value not in ATTRIBUTES[attr]:
            raise UnsupportedProgram(f"Unknown filter value: {attr}={value}")
        fields = dict(selection.fields)
        field = f"{selection.label}_{attr}"
        if field in fields:
            raise UnsupportedProgram(f"Repeated {attr} filter on {selection.label}")
        fields[field] = (Fun("v", self.ind, self.ptype(value, ["v"])), [selection.label])
        return _Selection(fields, selection.label, False, selection.unique_checks)

    @staticmethod
    def _merge_selections(left, right):
        fields = dict(left.fields)
        for label, value in right.fields.items():
            if label in fields and fields[label] is not value:
                raise UnsupportedProgram(f"Conflicting branches define field {label}")
            fields[label] = value
        checks = left.unique_checks + tuple(
            check for check in right.unique_checks if check not in left.unique_checks
        )
        return _Selection(fields, left.label, True, checks)

    def clevr_to_question_rec(self, question):
        """Compile a supported CLEVR functional-program DAG to a question record.

        Selection nodes build record types. ``query_*`` nodes produce attribute
        results which may either terminate the program or feed a terminal
        ``equal_*`` comparison of two independently compiled branches.
        Terminal ``count`` and ``exist`` summarize a non-unique selection.
        A count may also feed a terminal integer comparison.
        """
        program = question["program"]
        if not program:
            raise UnsupportedProgram("Empty program")
        labels = (f"x{i}" for i in count())
        results = {}
        for index, step in enumerate(program):
            name = step.get("function", step.get("type", ""))
            values = step.get("value_inputs", [])
            inputs = step.get("inputs")
            if not isinstance(inputs, list) or any(
                    not isinstance(source, int) or source < 0 or source >= index
                    for source in inputs):
                raise UnsupportedProgram(f"Invalid input indices at operation {index}: {inputs}")

            if name == "scene" and not inputs and not values:
                label = next(labels)
                results[index] = _Selection({label: self.ind}, label)
            elif name.startswith("filter_"):
                attr = name.removeprefix("filter_")
                if len(inputs) != 1 or not isinstance(results[inputs[0]], (_Selection, _UnionSelection)):
                    raise UnsupportedProgram(f"{name} requires one selection input")
                source = results[inputs[0]]
                if attr not in ATTRIBUTES or len(values) != 1 or values[0] not in ATTRIBUTES[attr]:
                    raise UnsupportedProgram(f"Invalid filter: {step}")
                if isinstance(source, _Selection):
                    if source.unique:
                        raise UnsupportedProgram(f"Invalid filter: {step}")
                    results[index] = self._filter_selection(source, attr, values[0])
                else:
                    results[index] = _UnionSelection(tuple(
                        self._filter_selection(branch, attr, values[0])
                        for branch in source.branches
                    ))
            elif name == "unique" and len(inputs) == 1 and not values:
                source = results[inputs[0]]
                if not isinstance(source, _Selection) or source.unique:
                    raise UnsupportedProgram("unique requires a non-unique selection")
                check = (self.record_type(dict(source.fields)), source.label)
                results[index] = _Selection(source.fields, source.label, True,
                                            source.unique_checks + (check,))
            elif name == "relate" and len(inputs) == 1:
                source = results[inputs[0]]
                if (not isinstance(source, _Selection) or not source.unique
                        or len(values) != 1 or values[0] not in RELATIONS):
                    raise UnsupportedProgram(f"Invalid relation: {step}")
                other = next(labels)
                fields = dict(source.fields)
                fields[other] = self.ind
                fields[f"{other}_{source.label}_{values[0]}"] = (
                    Fun("u", self.ind, Fun("v", self.ind, self.ptype(values[0], ["u", "v"]))),
                    [other, source.label],
                )
                results[index] = _Selection(fields, other, False, source.unique_checks)
            elif name in SAME_RELATIONS and len(inputs) == 1 and not values:
                source = results[inputs[0]]
                if not isinstance(source, _Selection) or not source.unique:
                    raise UnsupportedProgram(f"{name} requires a unique selection")
                reference = source.label
                other = next(labels)
                fields = dict(source.fields)
                fields[other] = self.ind
                fields[f"{other}_{reference}_{name}"] = (
                    Fun("u", self.ind, Fun("v", self.ind, self.ptype(name, ["u", "v"]))),
                    [other, reference],
                )
                fields[f"{other}_{reference}_distinct"] = (
                    Fun("u", self.ind, Fun("v", self.ind, self.ptype("distinct", ["u", "v"]))),
                    [other, reference],
                )
                results[index] = _Selection(fields, other, False, source.unique_checks)
            elif name == "union" and len(inputs) == 2 and not values:
                left, right = (results[source] for source in inputs)
                results[index] = _UnionSelection(
                    self._set_selection(left) + self._set_selection(right)
                )
            elif name.startswith("query_") and len(inputs) == 1 and not values:
                attr = name.removeprefix("query_")
                if attr not in ATTRIBUTES:
                    raise UnsupportedProgram(f"Unknown query attribute: {attr}")
                source = results[inputs[0]]
                if not isinstance(source, _Selection) or not source.unique:
                    raise UnsupportedProgram(f"{name} requires a unique selection")
                results[index] = _AttributeQuery(source, attr)
            elif name in ("count", "exist"):
                if len(inputs) != 1 or values:
                    raise UnsupportedProgram(f"{name} requires one selection input and no values")
                source = results[inputs[0]]
                if isinstance(source, _Selection) and source.unique:
                    raise UnsupportedProgram(f"{name} requires a non-unique selection")
                if not isinstance(source, (_Selection, _UnionSelection)):
                    raise UnsupportedProgram(f"{name} requires a selection input")
                if name == "count" and index != len(program) - 1:
                    results[index] = _CountQuery(source)
                elif index == len(program) - 1:
                    results[index] = self._aggregate_question_record(source, name)
                else:
                    raise UnsupportedProgram(f"Nonterminal {name} is not supported yet")
            elif name in ("less_than", "greater_than", "equal_integer"):
                if index != len(program) - 1 or len(inputs) != 2 or values:
                    raise UnsupportedProgram(f"{name} requires two count inputs and must be terminal")
                left, right = (results[source] for source in inputs)
                if not isinstance(left, _CountQuery) or not isinstance(right, _CountQuery):
                    raise UnsupportedProgram(f"{name} requires two count inputs")
                results[index] = self._count_comparison_record(name, left, right)
            elif name.startswith("equal_") and len(inputs) == 2 and not values:
                attr = name.removeprefix("equal_")
                if attr not in ATTRIBUTES:
                    raise UnsupportedProgram(f"Unknown comparison attribute: {attr}")
                left, right = (results[source] for source in inputs)
                if (not isinstance(left, _AttributeQuery)
                        or not isinstance(right, _AttributeQuery)
                        or left.attr != attr or right.attr != attr):
                    raise UnsupportedProgram(f"{name} requires two query_{attr} inputs")
                selection = self._merge_selections(left.selection, right.selection)
                results[index] = self._question_record(
                    selection,
                    CompareLookup(self, attr, left.selection.label, right.selection.label),
                    "The paper-aligned intrp and negative-answer semantics for attribute "
                    "comparisons are not yet specified; this is an operational answer.",
                )
            else:
                raise UnsupportedProgram(f"Unsupported or misplaced operation: {name}")

        result = results[len(program) - 1]
        if isinstance(result, _AttributeQuery):
            return self._attribute_question_record(result)
        if isinstance(result, Rec):
            return result
        raise UnsupportedProgram("Program must end with a query, comparison, count, or exist")

    def answer(self, question):
        """Return an answer label and its evidence.

        Attribute queries/comparisons return a single Rec. Aggregates and count
        comparisons return lists of all branch witness records (possibly empty).
        Their classifiers deduplicate target individuals. Count labels are strings,
        matching CLEVR's answers. Presupposition checks run first.
        """
        question_record = self.clevr_to_question_rec(question)
        for background, label in question_record["unique_checks"]:
            referents = {take[label] for take in self.build_witness_takes(background)}
            if len(referents) != 1:
                raise AmbiguousReference(f"unique({label}) found {len(referents)} objects")
        if "aggregate" in question_record:
            if "branches" in question_record:
                takes, projected, seen = [], [], set()
                for branch in question_record["branches"]:
                    for take in self.build_witness_takes(branch["background"]):
                        target = take[branch["label"]]
                        if target in seen:
                            continue
                        seen.add(target)
                        takes.append(take)
                        projected.append(Rec({"target": target}))
                return question_record["clfr"].app(projected).comps.pred.name, takes
            takes = list(self.build_witness_takes(question_record["bg"]))
            return question_record["clfr"].app(takes).comps.pred.name, takes
        if "comparison" in question_record:
            takes = [
                take
                for branch in question_record["branches"]
                for take in self.build_witness_takes(branch["background"])
            ]
            return question_record["clfr"].app(takes).comps.pred.name, takes
        takes = list(self.build_witness_takes(question_record["bg"]))
        if len(takes) != 1:
            raise AmbiguousReference(f"Expected one witness; found {len(takes)}")
        if "intrp" in question_record:
            answer_predicate = question_record["clfr"].appc(takes[0])
            answer_type = question_record["intrp"].appc(takes[0]).appc(answer_predicate)
            return answer_type.comps.pred.name, takes[0]
        return question_record["clfr"].app(takes[0]).comps.pred.name, takes[0]

    def description_type(self, **attributes):
        """Compose a one-entity description, e.g. a small red cube, as a record type."""
        fields = {"x": self.ind}
        for attr, value in attributes.items():
            if attr not in ATTRIBUTES or value not in ATTRIBUTES[attr]:
                raise ValueError(f"Unknown CLEVR attribute: {attr}={value}")
            fields[f"x_{attr}"] = (Fun("v", self.ind, self.ptype(value, ["v"])), ["x"])
        return self.record_type(fields)

    def matches(self, **attributes):
        return next(self.build_witness_takes(self.description_type(**attributes)), None) is not None
