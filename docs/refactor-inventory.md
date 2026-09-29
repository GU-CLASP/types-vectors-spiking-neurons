# Refactor inventory — 2026-09-23

Before the refactor, this repository had **no modifications to tracked files**.
Its starting commit was `b01fbae` (`remove unused notes`). Untracked files are
listed separately below. `archive/` is ignored and must not be assumed to have
Git backups. It currently occupies about 118 MiB.

## Preserved and migrated

| Original | Active replacement | Original preserved locally |
| --- | --- | --- |
| `ttr_clevr.py` | `src/spinls/clevr.py`: scene conversion, question records, witness search | `archive/originals/ttr_clevr.py` |
| `ttr_clevr-demo.py` | `examples/clevr_ground_truth.py`, with optional LaTeX snippet export | `archive/originals/ttr_clevr-demo.py` |
| `ttr_clevr-run.py` | `examples/clevr_evaluate.py`: explicit supported/unsupported counts | `archive/originals/ttr_clevr-run.py` |
| `util.py` | `CLEVR` loader in `src/spinls/clevr.py`; removed report-only pretty-printer | `archive/originals/util.py` |
| `cifar10.py` | `src/spinls/cifar10.py`; retained hierarchical dataset helpers | `archive/originals/cifar10.py` |
| `ttr-hdc-glue_cifar10.ipynb` | `notebooks/`, with new TTR bridge and cleared outputs | `archive/originals/ttr-hdc-glue_cifar10.ipynb` (includes old outputs) |
| `train_cifar_classifiers.py` evaluation helper | `src/spinls/evaluation.py` | Whole script in `archive/experiments/` |
| `.nix/` | Retained existing environment, including server configuration | Unchanged |
| `data/` | Existing local CLEVR/CIFAR symlinks retained and ignored | Targets not moved or altered |

Migration fixes include `.comps.items()` instead of `.__dict__`, querying the
requested attribute instead of always color, correct predicate arguments in
answers, deterministic non-mutating witness search, dependent constraints in
partial checks, explicit program input validation, and isolation of cached
judgements across scene snapshots. The old search added arbitrary unused
scene objects to witness records; this is removed.

The notebook had a stray syntax-breaking dot, a consensus helper that failed
on odd inputs, an evaluation function ignoring its model argument, and an
accumulator helper that mutated training totals during tie breaking. These
were fixed alongside imports, paths, and the TTR addition. It remains a flat
single-label VSA experiment; the introductory hierarchical proposal was not
an implemented feature.

## Supplanted or archived tracked content

| Archived path | Reason |
| --- | --- |
| `archive/experiments/hdc_glue.py` | Incomplete duplicate evaluator referring to undefined globals; notebook contains the working experiment |
| `archive/experiments/train_cifar_classifiers.py` | Standalone training detour, not needed by WP1 or notebook after extracting evaluation; original training loop also omitted gradient reset |
| `archive/reports/clevr.tex`, `clevr.pdf` | Generated report artifacts; optional small LaTeX output replaces them |
| `archive/reports/json_listing.tex`, `vsa-ttr.bib` | Report-specific formatting and bibliography, not active Python dependencies |
| `archive/originals/.gitmodules` | Stale `code/pyttr` entry; no corresponding tracked submodule |

## Untracked material: review before deleting

**These were not committed in this repository at the start of the refactor.**

| Original untracked path | Current location | Notes |
| --- | --- | --- |
| `ttr_clevr-find_examples.py` | `archive/experiments/ttr_clevr-find_examples.py` | Exploratory question selection and debugger stop; may contain useful selection criteria |
| `ttrspa.py` | `archive/experiments/ttrspa.py` | Early SPA/CLEVR scene experiment; overlapping direction now exists in sibling `pyttr2/src/pyttr/spa/` |
| `makefile` | `archive/reports/makefile` | Old LaTeX target referenced missing `clevr-ttr.py`; replaced by new `Makefile` |
| `notes/PaM-1.pdf` | `archive/artifacts/notes/PaM-1.pdf` | Untracked research PDF; no claim that it is obsolete |
| `models/vgg16_cifar10_flat/model.pt` | `archive/artifacts/models/vgg16_cifar10_flat/model.pt` | About 59 MiB; no Git copy |
| `models/vgg16_cifar10_flat/train_log.txt` | Same directory under `archive/artifacts/models/` | Untracked training log |
| `models/vgg16_cifar10_hierarchical/model.pt` | `archive/artifacts/models/vgg16_cifar10_hierarchical/model.pt` | About 59 MiB; no Git copy |
| `models/vgg16_cifar10_hierarchical/train_log.txt` | Same directory under `archive/artifacts/models/` | Untracked training log |
| `.ipynb_checkpoints/` (previously ignored) | `archive/artifacts/notebook-checkpoints/` | Four local checkpoints, including an early HD-Glue Python script with a TTR record example; no Git recovery guarantee |
| `data/CLEVR_v1.0` | Unchanged | Symlink to `/home/bill/Research/Data/CLEVR_v1.0` |
| `data/cifar-10-batches-py` | Unchanged | Symlink to `/home/bill/Research/Data/cifar-10-batches-py` |

Ignored LaTeX auxiliary files (`clevr.aux`, `.fdb_latexmk`, `.fls`, `.log`) were
also moved to `archive/reports/`. Old root bytecode moved to
`archive/artifacts/root-bytecode/`. The existing `.direnv/` environment remains
in place. Notebook checkpoints include distinct old experimental content and
should be reviewed before deleting the archive.

No datasets or weights were downloaded, no training was run, and no files were
changed in the sibling PyTTR repository. Its pre-existing modified notebook
`../pyttr2/notebooks/spa-ttr.ipynb` was left in place.

## Verification

- All 13 removed tracked files were compared with `git show HEAD:<path>`;
  each has a byte-identical original in `archive/`.
- `make check`: 13 tests passed, including synthetic HD-Glue/TTR integration.
- Both offline demos passed; live imports resolve to `../pyttr2/src/pyttr`.
- Original validation question 115673 still answers `green` correctly;
  its LaTeX snippet exports successfully.
- The first 20 supported validation questions all matched their dataset
  answers; 116 unsupported programs were skipped on the way to that sample.
- A real, untrained Faster R-CNN forward pass on a 64×64 blank image passed
  through the adapter without downloading detector or backbone weights.
- Active Python files parse, the notebook validates against its schema, and
  `git diff --check` passes.

Full CIFAR notebook execution/accuracy, trained CLEVR perception, editable
installation in a fresh environment, and rebuilding Nix were not verified.
