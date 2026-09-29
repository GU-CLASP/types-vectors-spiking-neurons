"""Run with `make demo`, or answer a random scene's questions from a local dataset.

PYTHONPATH=src:../pyttr2/src python examples/clevr_ground_truth.py \
    --data-dir data/CLEVR_v1.0
"""
import argparse
import json
from pathlib import Path
import random

from spinls.clevr import AmbiguousReference, CLEVR, CLEVRModel, UnsupportedProgram, clevr_to_h_scene
from spinls.fixtures import tiny_question, tiny_scene


def witness_assignment(model, question_record, take):
    """Show only the individual variables, omitting redundant evidence fields."""
    if "branches" in question_record:
        labels = [
            label
            for branch in question_record["branches"]
            for label in model.individual_labels(branch["background"])
            if label in take
        ]
        return ", ".join(f"{label} = {take[label]}" for label in dict.fromkeys(labels))
    return ", ".join(
        f"{label} = {take[label]}"
        for label in model.individual_labels(question_record["bg"])
    )


def print_question_representation(question_record):
    """Print the compiled question without repeating PyTTR function domains."""
    print("Question representation:")
    if "meaning_type" in question_record:
        print("  Question meaning type:", question_record["meaning_type"].show())
    print("  Background:", question_record["bg"].show())
    checks = question_record["unique_checks"]
    if checks:
        print("  Presuppositions:")
        for background, label in checks:
            print(f"    unique({label}): {background.show()}")
    if "intrp" in question_record:
        print("  Interpretation:", question_record["intrp"].body.show())
    print("  Answer classifier:", question_record["clfr"].body.show())
    if "semantic_note" in question_record:
        print("  Semantic limitation:", question_record["semantic_note"])
    if "aggregate" in question_record:
        print("  Aggregation:", question_record["aggregate"])
    if "comparison" in question_record:
        print("  Integer comparison:", question_record["comparison"])
    if "branches" in question_record:
        print("  Witness branches:")
        for branch in question_record["branches"]:
            side = f'{branch["side"]} ' if "side" in branch else ""
            print(f'    {side}target {branch["label"]}: {branch["background"].show()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--question-index", type=int, help="Run just this question instead")
    selection.add_argument("--scene-index", type=int, help="Run all questions for this image index")
    parser.add_argument("--seed", type=int, help="Reproduce random scene selection")
    parser.add_argument("--latex", type=Path, help="Write TTR snippets for all answered questions")
    args = parser.parse_args()
    if args.data_dir:
        dataset = CLEVR(args.data_dir)
        if args.question_index is not None:
            questions = [q for q in dataset.questions if q["question_index"] == args.question_index]
            if not questions:
                parser.error(f"No question with index {args.question_index}")
            scene_index = questions[0]["image_index"]
        else:
            # Choose uniformly over scenes with questions, not over questions.
            scene_indices = sorted({q["image_index"] for q in dataset.questions} & dataset.scenes.keys())
            if not scene_indices:
                parser.error("No scenes with questions found")
            scene_index = args.scene_index if args.scene_index is not None else random.Random(args.seed).choice(scene_indices)
            if scene_index not in scene_indices:
                parser.error(f"No scene with questions at image index {scene_index}")
            questions = [q for q in dataset.questions if q["image_index"] == scene_index]
        scene = dataset.scenes[scene_index]
    else:
        if args.question_index is not None or args.scene_index is not None or args.seed is not None:
            parser.error("Scene/question selection requires --data-dir")
        questions, scene = [tiny_question()], tiny_scene()
    model = CLEVRModel(clevr_to_h_scene(scene))
    print(f'Scene: {scene["image_index"]} ({len(questions)} questions)', flush=True)
    print("Ground-truth scene:")
    print(json.dumps(scene, indent=2, sort_keys=True))
    latex = []
    for question in questions:
        print(f'\nQuestion {question.get("question_index", "(synthetic)")}: {question["question"]}', flush=True)
        if "answer" in question:
            print("Dataset answer:", question["answer"])
        try:
            answer, take = model.answer(question)
        except UnsupportedProgram as exc:
            print("Unsupported:", exc)
            continue
        except AmbiguousReference as exc:
            print("No unique answer:", exc)
            continue
        question_rec = model.clevr_to_question_rec(question)
        print_question_representation(question_rec)
        if isinstance(take, list):
            print("Witness assignments:")
            if not take:
                print("  (no matching objects)")
            for record in take:
                print(" ", witness_assignment(model, question_rec, record))
        else:
            print("Witness assignment:", witness_assignment(model, question_rec, take))
        if "intrp" in question_rec:
            answer_predicate = question_rec["clfr"].appc(take)
            answer_type = question_rec["intrp"].appc(take).appc(answer_predicate)
            print("Classifier answer:", answer_predicate.show())
            print("Interpreted answer:", answer_type.show())
        print("Answer:", answer, flush=True)
        if args.latex:
            latex.append(f'% Question {question.get("question_index", "synthetic")}')
            records = take if isinstance(take, list) else [take]
            if not records:
                latex.append("% No matching objects; the witness list is empty.")
            items = [question_rec["bg"], *records]
            if "intrp" in question_rec:
                predicate = question_rec["clfr"].appc(take)
                items.extend((predicate, question_rec["intrp"].appc(take).appc(predicate)))
            else:
                items.append(question_rec["clfr"].app(take))
            latex.extend("\\[\n" + item.to_latex(vars=[]) + "\n\\]" for item in items)
    if args.latex:
        args.latex.write_text("\n\n".join(latex) + "\n")


if __name__ == "__main__":
    main()
