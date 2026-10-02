"""Browser comparison of CLEVR ground truth and learned visual perception."""

from collections.abc import Mapping
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
import random
from urllib.parse import parse_qs, urlencode, urlsplit

from pyttr.categorical import Rec, to_latex

from .clevr import AmbiguousReference, CLEVR, CLEVRModel, UnsupportedProgram, clevr_to_h_scene
from .visual_backend import VisualBackendUnavailable, categorical_predictions


class HDataReference:
    """A compact, display-only reference to an H-data item."""

    def __init__(self, representation, object_id):
        self.representation = representation
        self.object_id = object_id

    def show(self):
        return f"#{self.representation}({self.object_id})"

    def to_latex(self, vars):
        return rf"\#\mathrm{{{self.representation}}}(\text{{{self.object_id}}})"


class CLEVRWebDemo:
    """Prepare scene, question, and TTR data for the web page."""

    def __init__(self, data_dir, split="val", *, visual_backend=None, paper_path=None):
        self.dataset = CLEVR(data_dir, split=split)
        self.visual_backend = visual_backend
        self.paper_path = Path(paper_path).resolve() if paper_path else None
        self.questions_by_scene = {}
        for question in self.dataset.questions:
            scene_index = question["image_index"]
            if scene_index in self.dataset.scenes:
                self.questions_by_scene.setdefault(scene_index, []).append(question)
        if not self.questions_by_scene:
            raise ValueError(f"No {split} scenes with associated questions")
        self.scene_indices = sorted(self.questions_by_scene)

    def select(self, scene_index=None, question_index=None):
        if scene_index is None:
            scene_index = random.choice(self.scene_indices)
        if scene_index not in self.questions_by_scene:
            raise KeyError(f"Unknown scene index: {scene_index}")
        questions = self.questions_by_scene[scene_index]
        if question_index is None:
            question = questions[0]
        else:
            question = next(
                (item for item in questions if item["question_index"] == question_index), None
            )
            if question is None:
                raise KeyError(f"Question {question_index} does not belong to scene {scene_index}")
        return self.dataset.scenes[scene_index], questions, question

    @staticmethod
    def _uses_spatial_relations(question):
        return any(step.get("function") == "relate" for step in question["program"])

    def evaluate(self, scene, question, answer_requested=False, backend="ground-truth"):
        if backend not in ("ground-truth", "visual"):
            raise KeyError(f"Unknown backend: {backend}")
        predictions = ()
        representation = "json"
        backend_error = None
        if backend == "ground-truth":
            model = CLEVRModel(clevr_to_h_scene(scene))
        else:
            representation = "vector"
            if self.visual_backend is None:
                model = CLEVRModel({})
                backend_error = "The visual backend is not configured."
            else:
                try:
                    observation = self.visual_backend.observe(
                        self.dataset.get_image_path(scene), scene["image_index"]
                    )
                    model = observation.model
                    predictions = observation.predictions
                except VisualBackendUnavailable as exc:
                    model = CLEVRModel({})
                    backend_error = str(exc)
            if backend_error is None and self._uses_spatial_relations(question):
                backend_error = (
                    "This question needs a learned spatial-relation classifier. "
                    "The detector and attribute heads are available, but relation training is pending."
                )
        try:
            question_record = model.clevr_to_question_rec(question)
        except UnsupportedProgram as exc:
            return {"error": str(exc), "answer": None, "take_latex": None,
                    "meaning_type_latex": None, "answer_semantics_latex": None,
                    "semantic_note": None, "predictions": predictions,
                    "representation": representation, "backend": backend}

        # Keep operational metadata such as unique_checks and branches outside
        # the displayed semantic record.
        semantic_record = Rec({
            label: question_record[label]
            for label in ("bg", "intrp", "clfr")
            if label in question_record
        })
        result = {
            "error": backend_error,
            "question_latex": semantic_record.to_latex(vars=[]),
            "meaning_type_latex": (
                question_record["meaning_type"].to_latex(vars=[])
                if "meaning_type" in question_record else None
            ),
            "semantic_note": question_record.get("semantic_note"),
            "answer": None,
            "take_latex": None,
            "answer_semantics_latex": None,
            "predictions": predictions,
            "representation": representation,
            "backend": backend,
        }
        if not answer_requested or backend_error:
            return result
        try:
            answer, take = model.answer(question)
        except (UnsupportedProgram, AmbiguousReference) as exc:
            result["error"] = str(exc)
            return result
        result["answer"] = answer
        result["take_latex"] = to_latex(
            self._display_take(question_record, take, representation=representation), vars=[]
        )
        if "intrp" in question_record:
            answer_predicate = question_record["clfr"].appc(take)
            answer_type = question_record["intrp"].appc(take).appc(answer_predicate)
            result["answer_semantics_latex"] = (
                r"\begin{aligned}"
                r"q.\mathrm{clfr}(s)&=" + to_latex(answer_predicate, vars=[]) + r"\\"
                r"q.\mathrm{intrp}(s)(q.\mathrm{clfr}(s))&="
                + to_latex(answer_type, vars=[]) + r"\end{aligned}"
            )
        return result

    @staticmethod
    def _display_take(question_record, take, representation="json"):
        """Project to relevant fields and abbreviate each H-data payload."""
        if "branches" in question_record:
            labels = list(dict.fromkeys(
                label
                for branch in question_record["branches"]
                for label in branch["background"].comps
            ))
        else:
            labels = list(question_record["bg"].comps)

        def abbreviate(value):
            if isinstance(value, tuple):
                return tuple(
                    HDataReference(representation, item["id"])
                    if isinstance(item, (Mapping, Rec)) and item.get("id") is not None
                    else item
                    for item in value
                )
            return value

        def display_record(record):
            return Rec({
                label: abbreviate(record[label])
                for label in labels
                if label in record
            })

        return (
            [display_record(record) for record in take]
            if isinstance(take, list) else display_record(take)
        )

    def image_path(self, scene_index):
        scene = self.dataset.scenes[scene_index]
        path = self.dataset.get_image_path(scene).resolve()
        root = self.dataset.clevr_dir.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _ground_truth_overlays(scene):
        return "\n".join(
            f'<span class="object-id" data-x="{float(obj["pixel_coords"][0])}" '
            f'data-y="{float(obj["pixel_coords"][1])}">{scene["image_index"]}-{index}</span>'
            for index, obj in enumerate(scene["objects"])
            if len(obj.get("pixel_coords", ())) >= 2
        )

    @staticmethod
    def _visual_overlays(predictions):
        return "\n".join(
            '<span class="detection-box" '
            f'data-x1="{prediction.box[0]}" data-y1="{prediction.box[1]}" '
            f'data-x2="{prediction.box[2]}" data-y2="{prediction.box[3]}">'
            f'<span>{escape(prediction.object_id)}</span></span>'
            for prediction in predictions
        )

    @staticmethod
    def _visual_details(predictions):
        if not predictions:
            return '<p class="empty">No objects were detected at the configured score threshold.</p>'
        rows = []
        for prediction in predictions:
            labels = categorical_predictions(prediction)
            attributes = " ".join(
                f'<span><b>{escape(label)}</b> {confidence:.0%}</span>'
                for label, confidence in labels.values()
            )
            rows.append(
                '<div class="prediction-row">'
                f'<code>{escape(prediction.object_id)}</code>'
                f'<span class="detector-score">object {prediction.detection_score:.0%}</span>'
                f'<span class="prediction-attrs">{attributes}</span>'
                '</div>'
            )
        return "".join(rows)

    def render_info(self):
        metadata = self.visual_backend.checkpoint_metadata() if self.visual_backend else {}
        detector = metadata.get("detector") or {}
        attributes = metadata.get("attributes") or {}
        detector_metrics = detector.get("validation") or {}
        attribute_metrics = attributes.get("best_validation") or {}

        def percentage(value):
            return f"{value:.2%}" if isinstance(value, (int, float)) else "pending"

        detector_rows = "".join(
            f"<tr><th>{label}</th><td>{percentage(detector_metrics.get(key))}</td></tr>"
            for key, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))
        )
        attribute_rows = "".join(
            f"<tr><th>{name.title()}</th><td>{percentage(attribute_metrics.get(name))}</td></tr>"
            for name in ("color", "size", "material", "shape", "macro")
        )
        paper_link = (
            '<a href="/paper">project formalization manuscript</a>'
            if self.paper_path and self.paper_path.is_file()
            else 'project formalization manuscript (<code>../overleaf/spinls-overleaf/naloma.tex</code>)'
        )
        source = "https://github.com/GU-CLASP/types-vectors-spiking-neurons/blob/main/code"
        return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SPINLS · How the demo works</title><style>{STYLES}</style></head>
<body><main class="info-page">
  <header><div><span class="eyebrow">SPINLS</span><h1>How the demo works</h1></div><a class="quiet-button" href="/">Back to demo</a></header>
  <article class="card prose">
    <h2>Two views of one question</h2>
    <p>The ground-truth tab treats CLEVR's annotated scene graph as H-data. The model tab starts from image pixels, uses Faster R-CNN to individuate objects, and applies learned categorical heads to box-head feature vectors. Both paths compile the same CLEVR functional program into the same TTR question representation and run the same witness-take search. Keeping the scene and question fixed makes disagreements traceable to perception.</p>
    <div class="pipeline"><span>image / scene graph</span><b>→</b><span>individuated H-data</span><b>→</b><span>TTR witness take</span><b>→</b><span>answer classifier</span></div>

    <h2>TTR and H-data</h2>
    <p>The formal account distinguishes low-level perceptual data from individuated high-level data (H-data). An individual maps to an H-data item through the hash operation, and tuples of those items provide evidence for predicate types. A question contributes a background record type, an interpretation, and a classifier; witness search assigns perceived individuals to the record fields and checks classifier-backed ptypes. See the {paper_link}, especially “Low-level and high-level perceptual data”, “Perceptual data as evidence for ptypes”, “Definition of witness takes”, and “Adapting the general account to the CLEVR dataset”.</p>
    <p>Implementation: <a href="{source}/src/spinls/clevr.py">CLEVR-to-TTR semantics and witness search</a>, <a href="https://github.com/robincooper/pyttr2">PyTTR</a>, and <a href="{source}/src/spinls/web.py">this comparison UI</a>.</p>

    <h2>Neural perception</h2>
    <p>The object detector is torchvision Faster R-CNN with a ResNet-50 feature-pyramid backbone. It starts from the COCO-pretrained <code>COCO_V1</code> weights, replaces the COCO predictor with background/entity outputs, and is fine-tuned on CLEVR at 320×480 pixels. Training targets are approximate boxes projected from CLEVR scene annotations, so detector evaluation measures agreement with those pseudo-boxes.</p>
    <p>For attributes, detected boxes are encoded by the COCO-pretrained Faster R-CNN ROI pool and box head. Four independently trained softmax heads predict the mutually exclusive CLEVR families: color, size, material, and shape. The TTR classifiers use each family's top prediction. Spatial-relation classifiers are still pending; the model tab reports that limitation instead of consulting ground-truth relations.</p>
    <p>Implementation: <a href="{source}/src/spinls/detection_training.py">detector training and metrics</a>, <a href="{source}/src/spinls/perception.py">ROI features and TTR adapters</a>, <a href="{source}/src/spinls/vision_training.py">attribute-head training</a>, and <a href="{source}/src/spinls/visual_backend.py">live visual backend</a>.</p>

    <h2>Current evaluation</h2>
    <div class="metric-grid"><table><caption>Detector · validation</caption>{detector_rows}</table><table><caption>Attributes · validation top-1</caption>{attribute_rows}</table></div>
    <p class="notation-note">Missing detector values appear as “pending” until <code>artifacts/clevr-detector/detector.pt</code> is copied from the training server. Relation evaluation will be added with the relation model.</p>

    <h2>References</h2>
    <ul>
      <li>Ren, He, Girshick &amp; Sun (2015), <a href="https://arxiv.org/abs/1506.01497">Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks</a>.</li>
      <li>Lin et al. (2014), <a href="https://arxiv.org/abs/1405.0312">Microsoft COCO: Common Objects in Context</a>.</li>
      <li>Johnson et al. (2017), <a href="https://arxiv.org/abs/1612.06890">CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning</a>.</li>
      <li>Cooper &amp; Ginzburg (2015), “Type Theory with Records for Natural Language Semantics”.</li>
      <li>Cooper (2023), <a href="https://global.oup.com/academic/product/from-perception-to-communication-9780192871312">From Perception to Communication</a>.</li>
      <li><a href="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html">Torchvision Faster R-CNN ResNet50-FPN documentation</a>.</li>
      <li>Approximate CLEVR box heuristic: <a href="https://github.com/larchen/clevr-vqa/blob/a224099addec82cf25f21d1fcbe11b15d3c02355/bounding_box.py">Larry Chen's immutable source revision</a>.</li>
    </ul>
  </article>
</main></body></html>"""

    def render(self, scene_index=None, question_index=None, answer_requested=False,
               backend="ground-truth"):
        scene, questions, question = self.select(scene_index, question_index)
        result = self.evaluate(scene, question, answer_requested, backend=backend)
        selected_index = question["question_index"]
        query_values = {
            "scene": scene["image_index"],
            "question": selected_index,
            "backend": backend,
        }
        query = urlencode(query_values)
        ground_truth_query = urlencode({**query_values, "backend": "ground-truth"})
        visual_query = urlencode({**query_values, "backend": "visual"})
        query_html = escape(query, quote=True)
        ground_truth_query_html = escape(ground_truth_query, quote=True)
        visual_query_html = escape(visual_query, quote=True)
        options = "\n".join(
            f'<option value="{item["question_index"]}"'
            f'{" selected" if item is question else ""}>'
            f'{escape(str(item["question_index"]))} · {escape(item["question"])}</option>'
            for item in questions
        )
        visual = backend == "visual"
        overlays = (
            self._visual_overlays(result["predictions"])
            if visual else self._ground_truth_overlays(scene)
        )
        question_panel = (
            f'<div class="math scroll">\\[{escape(result["question_latex"])}\\]</div>'
            if result.get("question_latex")
            else '<p class="notice">The question could not be compiled.</p>'
        )
        backend_notice = (
            f'<p class="notice backend-notice">{escape(result["error"])}</p>'
            if result["error"] else ""
        )
        meaning_type_panel = (
            '<h3>Question meaning type</h3>'
            f'<div class="math scroll compact-math">\\[{escape(result["meaning_type_latex"])}\\]</div>'
            if result.get("meaning_type_latex") else ""
        )
        semantic_note = (
            f'<p class="notice semantic-note">{escape(result["semantic_note"])}</p>'
            if result.get("semantic_note") else ""
        )
        answer_panel = "<p class=\"empty\">Generate an answer to inspect its evidence.</p>"
        if answer_requested and result["error"]:
            answer_panel = f'<p class="notice">{escape(result["error"])}</p>'
        elif result["answer"] is not None:
            answer_semantics = (
                f'<h3>Answer composition</h3><div class="math scroll compact-math">'
                f'\\[{escape(result["answer_semantics_latex"])}\\]</div>'
                if result.get("answer_semantics_latex") else ""
            )
            answer_panel = (
                f'<div class="math scroll">\\[{escape(result["take_latex"])}\\]</div>'
                f'<p class="notation-note"><code>#{result["representation"]}(id)</code> '
                f'abbreviates the corresponding {"feature-vector" if visual else "CLEVR scene-graph"} '
                'H-data item; angle brackets denote evidence tuples.</p>'
                f'{answer_semantics}'
                '<div class="answer-row">'
                f'<span><small>Answer</small>{escape(str(result["answer"]))}</span>'
                f'<span><small>Dataset</small>{escape(str(question.get("answer", "—")))}</span>'
                '</div>'
            )
        disabled = " disabled" if result["error"] else ""
        labels_hidden = "" if visual else " hidden"
        labels_checked = " checked" if visual else ""
        overlay_label = "Detections" if visual else "Object IDs"
        perception_title = "Model predictions" if visual else "Ground truth"
        prediction_panel = (
            '<section class="prediction-panel"><h3>Detected objects and attributes</h3>'
            f'{self._visual_details(result["predictions"])}</section>'
            if visual else ""
        )
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SPINLS · CLEVR</title>
  <style>{STYLES}</style>
  <script>window.MathJax={{tex:{{displayMath:[["\\\\[","\\\\]"]]}}}};</script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
  <main>
    <header>
      <div><span class="eyebrow">SPINLS</span><h1>CLEVR perception and TTR</h1></div>
      <div class="scene-actions">
        <form class="scene-picker" method="get" action="/">
          <input type="hidden" name="backend" value="{backend}">
          <label for="scene">Image</label>
          <input id="scene" name="scene" type="number" min="0" step="1" value="{scene["image_index"]}" aria-label="CLEVR image index">
          <button type="submit">Go</button>
        </form>
        <a class="quiet-button" href="/?backend={backend}">Random scene</a>
        <a class="help-button" href="/info" aria-label="How this demo works">?</a>
      </div>
    </header>
    <nav class="backend-tabs" aria-label="Perception source">
      <a class="{'active' if not visual else ''}" href="/?{ground_truth_query_html}">Ground truth</a>
      <a class="{'active' if visual else ''}" href="/?{visual_query_html}">Model predictions</a>
    </nav>
    <section class="workspace">
      <article class="card scene-card">
        <div class="card-heading"><div><span class="eyebrow">{perception_title}</span><h2>Scene {scene["image_index"]}</h2></div><label><input id="show-ids" type="checkbox"{labels_checked}> {overlay_label}</label></div>
        <div class="scene-canvas">
          <img id="scene-image" src="/image?scene={scene["image_index"]}" alt="CLEVR scene {scene["image_index"]}">
          <div id="object-labels"{labels_hidden}>{overlays}</div>
        </div>
        {prediction_panel}
      </article>
      <article class="card question-card">
        <form method="get" action="/">
          <input type="hidden" name="scene" value="{scene["image_index"]}">
          <input type="hidden" name="backend" value="{backend}">
          <label class="field-label" for="question">Question</label>
          <select id="question" name="question" onchange="this.form.submit()">{options}</select>
        </form>
        <p class="question-text">{escape(question["question"])}</p>
        <h2>TTR representation</h2>
        {meaning_type_panel}
        {question_panel}
        {semantic_note}
        {backend_notice}
        <form method="post" action="/answer?{query_html}">
          <button type="submit"{disabled}>Generate answer</button>
        </form>
      </article>
    </section>
    <section class="card evidence-card">
      <h2>Situation take</h2>
      {answer_panel}
    </section>
  </main>
  <script>
    const checkbox = document.querySelector('#show-ids');
    const labels = document.querySelector('#object-labels');
    const image = document.querySelector('#scene-image');
    checkbox.addEventListener('change', () => {{ labels.hidden = !checkbox.checked; }});
    function placeLabels() {{
      if (!image.naturalWidth || !image.naturalHeight) return;
      labels.querySelectorAll('.object-id').forEach(label => {{
        label.style.left = `${{100 * Number(label.dataset.x) / image.naturalWidth}}%`;
        label.style.top = `${{100 * Number(label.dataset.y) / image.naturalHeight}}%`;
      }});
      labels.querySelectorAll('.detection-box').forEach(box => {{
        const x1 = Number(box.dataset.x1), y1 = Number(box.dataset.y1);
        const x2 = Number(box.dataset.x2), y2 = Number(box.dataset.y2);
        box.style.left = `${{100 * x1 / image.naturalWidth}}%`;
        box.style.top = `${{100 * y1 / image.naturalHeight}}%`;
        box.style.width = `${{100 * (x2 - x1) / image.naturalWidth}}%`;
        box.style.height = `${{100 * (y2 - y1) / image.naturalHeight}}%`;
      }});
    }}
    image.addEventListener('load', placeLabels);
    if (image.complete) placeLabels();
  </script>
</body>
</html>"""


STYLES = """
:root { color-scheme: light; --ink:#18201d; --muted:#64706a; --line:#d8ded9; --paper:#fff; --wash:#f3f5f1; --accent:#176b51; }
* { box-sizing:border-box; }
body { margin:0; background:var(--wash); color:var(--ink); font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; }
main { width:min(1180px,calc(100% - 32px)); margin:40px auto; }
header,.card-heading,.answer-row { display:flex; align-items:center; justify-content:space-between; gap:20px; }
header { margin-bottom:18px; }
.scene-actions,.scene-picker { display:flex; align-items:center; gap:8px; }
.scene-picker input { width:96px; border:1px solid var(--line); border-radius:7px; background:white; padding:8px 9px; color:var(--ink); font:inherit; }
.scene-picker button { margin:0; }
h1,h2,p { margin-top:0; } h1 { margin-bottom:0; font-size:clamp(24px,4vw,36px); letter-spacing:-.035em; }
h2 { margin-bottom:14px; font-size:14px; letter-spacing:.02em; }
.card-heading h2 { margin:2px 0 14px; }
h3 { margin:16px 0 8px; color:var(--muted); font-size:12px; letter-spacing:.08em; text-transform:uppercase; }
.eyebrow,.field-label,small { display:block; color:var(--muted); font-size:11px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.backend-tabs { display:flex; gap:4px; margin:0 0 18px; border-bottom:1px solid var(--line); }
.backend-tabs a { margin-bottom:-1px; border:1px solid transparent; border-radius:8px 8px 0 0; padding:10px 15px; color:var(--muted); font-weight:650; text-decoration:none; }
.backend-tabs a.active { border-color:var(--line); border-bottom-color:var(--paper); background:var(--paper); color:var(--accent); }
.workspace { display:grid; grid-template-columns:minmax(0,1.1fr) minmax(340px,.9fr); gap:18px; align-items:start; }
.card { border:1px solid var(--line); border-radius:12px; background:var(--paper); box-shadow:0 8px 30px rgba(30,45,37,.045); padding:18px; }
.scene-canvas { position:relative; overflow:hidden; border-radius:8px; background:#e8ebe7; line-height:0; }
.scene-canvas img { display:block; width:100%; height:auto; }
#object-labels { position:absolute; inset:0; }
.object-id { position:absolute; transform:translate(-50%,-50%); padding:3px 6px; border:1px solid rgba(255,255,255,.75); border-radius:4px; background:rgba(18,25,22,.82); color:white; font:600 11px/1.2 ui-monospace,monospace; box-shadow:0 1px 5px #0005; }
.detection-box { position:absolute; border:2px solid #ffca57; box-shadow:0 0 0 1px #0005, inset 0 0 0 1px #0003; }
.detection-box > span { position:absolute; top:-21px; left:-2px; padding:3px 5px; border-radius:3px 3px 0 0; background:#ffca57; color:#31280f; font:700 10px/1.2 ui-monospace,monospace; }
label { color:var(--muted); font-size:13px; } input[type=checkbox] { accent-color:var(--accent); vertical-align:-1px; }
select { width:100%; margin:6px 0 14px; border:1px solid var(--line); border-radius:7px; background:white; padding:10px 34px 10px 10px; color:var(--ink); font:inherit; }
.question-text { min-height:46px; margin-bottom:22px; font-size:17px; line-height:1.45; }
.math { min-height:150px; padding:14px; border:1px solid var(--line); border-radius:8px; background:#fbfcfa; }
.compact-math { min-height:0; }
.scroll { overflow:auto; }
button,.quiet-button,.help-button { display:inline-block; border:1px solid var(--accent); border-radius:7px; padding:9px 14px; background:var(--accent); color:white; font:600 13px/1.2 inherit; text-decoration:none; cursor:pointer; }
.quiet-button { background:transparent; color:var(--accent); }
.help-button { display:grid; width:36px; height:36px; place-items:center; border-radius:50%; padding:0; font-size:16px; }
button { margin-top:16px; } button:disabled { cursor:not-allowed; opacity:.45; }
.prediction-panel { margin-top:16px; }
.prediction-row { display:grid; grid-template-columns:auto auto 1fr; gap:8px 12px; align-items:center; padding:9px 0; border-top:1px solid var(--line); }
.prediction-row code { font-weight:700; }
.detector-score { color:var(--muted); font-size:12px; }
.prediction-attrs { display:flex; justify-content:flex-end; flex-wrap:wrap; gap:6px; }
.prediction-attrs span { border-radius:999px; background:var(--wash); padding:3px 7px; color:var(--muted); font-size:11px; }
.prediction-attrs b { color:var(--ink); }
.evidence-card { margin-top:18px; }
.evidence-card .math { min-height:90px; }
.answer-row { justify-content:flex-start; margin-top:14px; gap:42px; font-size:24px; font-weight:650; }
.answer-row small { margin-bottom:2px; }
.empty,.notice { margin:0; color:var(--muted); }
.notice { padding:12px; border-radius:7px; background:#fff4e8; color:#85501b; }
.semantic-note,.backend-notice { margin-top:12px; }
.info-page { max-width:920px; }
.info-page header { align-items:flex-start; }
.prose { padding:clamp(22px,5vw,52px); }
.prose h2 { margin:34px 0 10px; font-size:21px; letter-spacing:-.015em; }
.prose h2:first-child { margin-top:0; }
.prose p,.prose li { max-width:78ch; }
.prose a { color:var(--accent); }
.pipeline { display:flex; align-items:center; justify-content:center; gap:10px; margin:24px 0; border-radius:8px; background:var(--wash); padding:18px; font-size:13px; text-align:center; }
.pipeline span { border:1px solid var(--line); border-radius:6px; background:white; padding:7px 9px; }
.metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
table { width:100%; border-collapse:collapse; border:1px solid var(--line); }
caption { padding:8px; background:var(--wash); font-weight:700; text-align:left; }
th,td { border-top:1px solid var(--line); padding:7px 9px; text-align:left; }
td { text-align:right; font-variant-numeric:tabular-nums; }
@media (max-width:800px) { main { margin:20px auto; } .workspace,.metric-grid { grid-template-columns:1fr; } header { align-items:flex-start; flex-direction:column; } .scene-actions,.pipeline { flex-wrap:wrap; } .prediction-row { grid-template-columns:auto 1fr; } .prediction-attrs { grid-column:1/-1; justify-content:flex-start; } }
"""


def make_handler(demo):
    class DemoHandler(BaseHTTPRequestHandler):
        def _parameters(self):
            query = parse_qs(urlsplit(self.path).query)
            try:
                scene = int(query["scene"][0]) if "scene" in query else None
                question = int(query["question"][0]) if "question" in query else None
            except (ValueError, IndexError) as exc:
                raise KeyError("Scene and question indices must be integers") from exc
            backend = query.get("backend", ["ground-truth"])[0]
            if backend not in ("ground-truth", "visual"):
                raise KeyError(f"Unknown backend: {backend}")
            return scene, question, backend

        def _write(self, status, content, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            route = urlsplit(self.path).path
            try:
                if route == "/info":
                    self._write(200, demo.render_info().encode(), "text/html; charset=utf-8")
                    return
                if route == "/paper" and demo.paper_path and demo.paper_path.is_file():
                    self._write(200, demo.paper_path.read_bytes(), "application/pdf")
                    return
                scene_index, question_index, backend = self._parameters()
                if route == "/":
                    body = demo.render(scene_index, question_index, backend=backend).encode()
                    self._write(200, body, "text/html; charset=utf-8")
                elif route == "/image" and scene_index is not None:
                    path = demo.image_path(scene_index)
                    self._write(200, path.read_bytes(), mimetypes.guess_type(path)[0] or "application/octet-stream")
                else:
                    self.send_error(404)
            except (KeyError, FileNotFoundError) as exc:
                self.send_error(404, str(exc))

        def do_POST(self):
            route = urlsplit(self.path).path
            if route != "/answer":
                self.send_error(404)
                return
            try:
                scene_index, question_index, backend = self._parameters()
                body = demo.render(
                    scene_index, question_index, answer_requested=True, backend=backend
                ).encode()
                self._write(200, body, "text/html; charset=utf-8")
            except KeyError as exc:
                self.send_error(404, str(exc))

    return DemoHandler


def serve(
    data_dir,
    split="val",
    host="127.0.0.1",
    port=8000,
    *,
    detector_checkpoint=None,
    attribute_checkpoint=None,
    device="cpu",
    detection_threshold=0.5,
    paper_path=None,
):
    visual_backend = None
    if detector_checkpoint is not None and attribute_checkpoint is not None:
        from .visual_backend import VisualCLEVRBackend

        visual_backend = VisualCLEVRBackend(
            detector_checkpoint,
            attribute_checkpoint,
            device=device,
            detection_threshold=detection_threshold,
        )
    demo = CLEVRWebDemo(
        Path(data_dir), split=split, visual_backend=visual_backend, paper_path=paper_path
    )
    server = ThreadingHTTPServer((host, port), make_handler(demo))
    print(f"CLEVR demo running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
