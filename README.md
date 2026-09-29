# SPINLS: perceptually grounded TTR

WP1 research code for developing and stress-testing PyTTR alongside
`../pyttr2`. The active CLEVR path uses categorical TTR directly. The retained
HD-Glue notebook is a separate VSA experiment.

## Working with the live PyTTR repository

Expected sibling layout:

```text
types-vectors-spiking-neurons/
├── pyttr2/                 # Its own Git repository; edit and commit here
│   └── src/pyttr/
└── code/                   # This Git repository
    ├── src/spinls/         # Grounding, perception, and CIFAR helpers
    ├── examples/           # Runnable ground-truth and perception demos
    ├── notebooks/         # Retained HD-Glue experiment
    ├── tests/             # Offline regression and integration checks
    ├── docs/              # Refactor inventory and WP1 boundaries
    ├── data/              # Ignored local dataset links (not downloaded)
    └── archive/           # Ignored originals, old experiments, local artifacts
```

With the existing Nix environment (or Python with NumPy and IPython installed),
**no package install is needed**:

```sh
make paths                 # Prints the exact working-tree import locations
make demo                  # Synthetic CLEVR question -> TTR -> answer
make vision-demo           # Synthetic perceptual evidence -> matching
make check                 # Offline tests; optional torch checks skip if absent
make notebook              # Requires JupyterLab and the notebook dependencies
```

The tracked Makefile prepends `src/` and `../pyttr2/src` to `PYTHONPATH` for its
commands. Override `PYTTR_SRC=/path/to/pyttr2/src` if needed. For arbitrary
commands use `PYTHONPATH=src:../pyttr2/src python ...` from this directory.

For a conventional Python environment, an editable install is another option:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

An editable install points at the **same Git working tree**, including
uncommitted changes; it does not create a second copy of PyTTR's source. Both
approaches let you commit and push PyTTR changes directly from `../pyttr2`.
Restart notebook kernels after changing imported PyTTR code. Commits remain
separate in the two repositories; neither approach pins a cross-repository
revision. Record both commit IDs when reporting an experiment.

The old `.gitmodules` entry pointed to an absent `code/pyttr` submodule and has
been archived. No symlink or submodule is required. The existing `.nix/`
environment is retained; it was not rebuilt as part of this refactor.

## CLEVR demos

`examples/clevr_ground_truth.py` replaces the old demo and can export a LaTeX
snippet with `--latex /tmp/clevr-example.tex`. With existing local data:

```sh
PYTHONPATH=src:../pyttr2/src python examples/clevr_ground_truth.py \
  --data-dir data/CLEVR_v1.0
PYTHONPATH=src:../pyttr2/src python examples/clevr_evaluate.py --limit 100
```

With a dataset, the demo selects a random scene and attempts every question
about it, printing the answer and witness record for each supported question.
Unsupported or ambiguous questions are reported without stopping the run.
Use `--seed 42` to reproduce the selection, `--scene-index N` to choose a scene,
or `--question-index N` to run one question. Without `--data-dir`, the small
synthetic demo still runs. `CLEVRModel.answer(question)` returns
`(answer_label, evidence)`: evidence is one record for attribute queries and
comparisons, or a list of all matching records for `count` and `exist` (including
an empty list for zero matches). Count comparisons also return their branch
witness records. Answers remain strings, including counts.

For a minimal browser interface over the same ground-truth model, run:

```sh
make web-demo
```

Then open `http://127.0.0.1:8000`. The page shows a random CLEVR scene,
optional ground-truth object-ID overlays, its associated questions, the
compiled TTR question, and—on request—the answer and situation take. Enter a
CLEVR image index in the header to open a specific scene. Use
`python examples/clevr_web.py --help` to select another dataset split, host,
or port. LaTeX is rendered by MathJax loaded from its public CDN.

For terminal `query_*` questions, the representation separates `clfr`, which
returns an elliptical attribute predicate, from `intrp`, which applies that
predicate to the focussed referent to construct the answer PType. The page
shows both stages. Count, existence, attribute-comparison, and integer-comparison
questions still use operational answer procedures; the demo marks their
paper-aligned interpretation and negative-answer semantics as unresolved.

The compiler supports `scene`, attribute `filter_*`, `unique`, `relate`,
`union`, and `query_*` programs. It follows the program's input graph, including two
independent attribute-query branches consumed by `equal_color`, `equal_shape`,
`equal_size`, or `equal_material`. It checks uniqueness before later filters
can hide ambiguity. Repeated filters for one attribute remain unsupported.
The four `same_*` operations are supported when their input is unique; they
introduce a distinct candidate that later filters, `unique`, and `query_*` can
consume. `union` takes the set union of two selection branches; subsequent
filters distribute over the branches, and overlapping objects are deduplicated
by terminal aggregation. Terminal `count` counts distinct objects in the
selection's target field, and `exist` returns `yes` if any match, otherwise
`no`. A nonterminal `count` can feed terminal `less_than`, `greater_than`, or
`equal_integer`; their two selections are evaluated independently before their
distinct target counts are compared. Empty selections are valid; a failed
earlier `unique` still raises a presupposition error. Set intersection remains
unsupported. This is not a complete CLEVR solver.

The perception demo composes controlled descriptions such as “a small red
cube” into record types and searches for witnesses. It includes both matching
and nonmatching descriptions; there is no natural-language parser yet.

To exercise the real Faster R-CNN architecture without downloading weights:

```sh
PYTHONPATH=src:../pyttr2/src OMP_NUM_THREADS=2 python examples/clevr_perception.py \
  --architecture-smoke
```

This uses random weights only to verify plumbing. `FasterRCNNDetector` accepts
an injected detector plus a crop-to-attribute-distributions callback. A trained
CLEVR detector/attribute classifier is still needed for real image matching.
See [WP1 boundaries](docs/wp1.md).

## HD-Glue notebook

`notebooks/ttr-hdc-glue_cifar10.ipynb` retains the dense-feature/VSA experiment,
uses the new categorical PyTTR API, and adds actual witness-condition and
record-type queries. It has a live-source setup cell and defaults to
`ALLOW_DOWNLOADS = False`. Full execution requires CIFAR-10 and a cached VGG16
hub repository/checkpoint, or explicit opt-in to fetching them. The offline
tests exercise its HD helpers and TTR bridge using synthetic vectors; they do
not measure CIFAR accuracy. Previous outputs are preserved only in the archive.

## Archive

See [the inventory](docs/refactor-inventory.md) **before deleting `archive/`**:
several scripts, a PDF, and about 117 MiB of checkpoints/logs were untracked.
They have local copies in the archive but no Git recovery guarantee. Nothing
has been committed or pushed by this refactor.
