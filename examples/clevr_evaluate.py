"""Evaluate the supported question fragment on an existing CLEVR dataset."""
import argparse
from collections import Counter
from pathlib import Path

from spinls.clevr import AmbiguousReference, CLEVR, CLEVRModel, UnsupportedProgram, clevr_to_h_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/CLEVR_v1.0"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=100, help="Maximum supported questions to evaluate")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    data = CLEVR(args.data_dir, args.split)
    counts = Counter()
    for question in data.questions:
        model = CLEVRModel(clevr_to_h_scene(data.scenes[question["image_index"]]))
        try:
            predicted, _take = model.answer(question)
        except UnsupportedProgram:
            counts["unsupported"] += 1
            continue
        except AmbiguousReference:
            counts["ambiguous_or_missing"] += 1
        else:
            counts["correct" if predicted == question["answer"] else "incorrect"] += 1
            if predicted != question["answer"]:
                print(question["question_index"], question["question"], predicted, question["answer"])
        counts["evaluated"] += 1
        if counts["evaluated"] >= args.limit:
            break
    print(dict(counts))


if __name__ == "__main__":
    main()
