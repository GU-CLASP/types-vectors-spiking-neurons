# WP1 boundaries and next experiments

The ground-truth and first neural-feature pipelines are:

```text
image -> detector -> crops -> attribute distributions
                                  |
                      external Boolean witness conditions
                                  |
controlled description -> dependent record type -> witness search -> match

image + oracle boxes -> Faster R-CNN box-head vectors -> attribute heads
                              |                           |
                         vector H-data       lazy Boolean witness conditions
```

`spinls.clevr.CLEVRModel` uses `pyttr.categorical` and one fresh `Possibility`
per scene snapshot. Each predicate's witness function returns a basic type
whose witness condition calls an external Python classifier. Evidence is
checked against the predicate's actual individual arguments. Positive
judgements are cached by PyTTR, so create a new model after changing evidence,
classifier weights, or thresholds; do not mutate an existing model's scene.

The ground-truth path and perceptual path share record-type construction and
witness search. Ground-truth metadata supplies exact attribute/relation
classifiers. Perceptual distributions supply strict-threshold categorical
decisions (probability > threshold); these are **not** probabilistic TTR
judgements or calibrated confidence in whole descriptions. Missing attributes
provide no positive evidence. A failed match means no witness was found, not
that the description's negation has been proved.

`Fun` already exists in current PyTTR. The demo uses object-language functions
for dependent fields and an `AttributeClassifier` Python-backed function body
for question answers. This provides a concrete test case for further function
development; it does not claim to implement linguistic modifiers or action
rules. The probabilistic and categorical APIs must remain distinct.

Terminal attribute questions now distinguish the classifier from the
interpretation. `clfr` returns an elliptical predicate meaning such as
`lambda x:Ind.blue(x)`, and `intrp` applies that predicate to the focussed
referent to construct `blue(s.x)`. A dependent record type constrains the
question-meaning record's `intrp` and `clfr` domains using its `bg` value.
The more general semantics proposed for other answer types is recorded in
`../formalisation/question-semantics-extension.md` but is not adopted here.

Terminal `count` and `exist` use `Fun` with a `ListType(background)` domain.
The Python answer workflow enumerates the complete scene-relative witness list;
the function body projects distinct target object IDs and computes a count or
existence answer. An empty list is valid evidence for this computation.
The list type checks its members, but does not certify completeness. Numerical
and yes/no predicate types here encode answer labels, not proofs of cardinality
or absence. Presupposition checks still run in Python before aggregation;
object-language presupposition failure remains an open question.
These aggregate answers remain operational: a paper-aligned `intrp`, complete
extension type, and justified negative-answer semantics have not been added.

`count` may also be an intermediate node consumed by terminal `less_than`,
`greater_than`, or `equal_integer`. The compiler retains each selected set as a
separate branch: it must not conjoin the two count backgrounds. The answer
workflow enumerates both branches, and the comparison function deduplicates
each branch's target IDs before returning a `yes`/`no` answer label.

`union` is represented by a categorical `JoinType` over the two branch record
types. Attribute filters distribute over its branches. For terminal aggregation,
the answer workflow obtains witnesses for each branch, projects each one to its
selected object ID, and deduplicates those IDs. The current fragment does not
support `unique`, relations, or attribute queries directly after a union;
CLEVR's union programs in this dataset use filters followed by `count`.

## Perceptual adapter

`FasterRCNNDetector.detect` expects float RGB `[3,H,W]` images in `[0,1]`, runs
in evaluation/inference mode, filters low-score detections, and classifies
each crop through an injected callback. `Detection` records carry boxes,
object IDs, scores, and distributions for any known CLEVR attributes.

[Torchvision Faster R-CNN](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)
returns boxes, class labels, and scores. Its standard pretrained labels are
COCO categories, not CLEVR attribute distributions. A CLEVR-trained detector
or objectness model plus attribute heads is therefore a future integration.
`build_untrained_faster_rcnn` disables both detector and backbone weights,
preventing implicit downloads. Random-weight output is not perceptual evidence
of CLEVR attributes; the architecture smoke supplies an empty attribute map.

The first vector-H-data milestone bypasses proposals and predictions.
`FasterRCNNFeatureExtractor` accepts externally supplied `OracleRegion` boxes
and runs them through the Faster R-CNN transform, backbone, RoI pool, and box
head. Each resulting `FeatureObject` stores a flat immutable tuple of reals.
`model_from_feature_objects` retains these vectors in the H-scene and calls an
injected feature classifier only when witness search requests an attribute;
the resulting distributions are cached once per object. This keeps the H-data
distinct from the categorical threshold decision.

`build_factorized_attribute_heads` constructs one Torch linear head for each
CLEVR attribute family, and `TorchAttributeClassifier` converts trained head
logits to distributions. `vision_training.py` defines the versioned tensor
cache, compatibility validation, factorized cross-entropy training, per-family
evaluation, best-checkpoint saving, and checkpoint loading. The repository
still needs mask-tight boxes for any later localization experiment; the current
experiment deliberately isolates attribute learning from localization.

`scene_object_to_box` supplies the initial no-render approximation from CLEVR's
pixel coordinates, 3D coordinates, camera-right direction, and shape-specific
projection constants. Its docstring retains an immutable link to the upstream
Apache-2.0 implementation. The port fixes the source's sequential coordinate
overwrite, scales to the loaded image size, and clips boxes to image bounds.
`examples/clevr_boxes.py` overlays IDs and ground-truth attributes for manual
quality checks. These regions are suitable as pseudo-oracle inputs for the
first attribute experiment, not yet as box-regression targets.

The perceptual stub deliberately supplies no spatial relations: ordering
2D bounding-box centers would not reproduce CLEVR's camera-relative 3D
relations. The ground-truth baseline retains the dataset relations.

## Next work

1. Run the resumable train/validation feature export and head training on the
   CUDA server, recording the cache manifests and validation metrics.
2. Calibrate per-family categorical thresholds from validation probabilities.
3. Connect the trained vector backend to attribute-only web questions and display its
   evidence as `#vector(object-id)`.
4. Replace oracle regions with learned proposals only after the attribute path
   has been validated independently.
5. Add a controlled description parser and compositional modifier examples
   that need `Fun`, beyond attribute conjunctions.
6. Specify action-rule inputs, licensed outputs, and agent state transitions;
   use these experiments to drive additions in `../pyttr2`.
7. Decide how classifier uncertainty should enter probabilistic TTR, rather
   than reusing categorical threshold decisions as probabilities.
8. Extend the CLEVR functional-program fragment only with corresponding
   semantics and dataset regressions (set intersection, integer comparisons).

The sibling PyTTR repository was inspected but not modified in this refactor.
Its existing uncommitted `notebooks/spa-ttr.ipynb` edit remains untouched.
