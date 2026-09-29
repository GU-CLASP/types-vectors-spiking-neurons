"""A small, dependency-free web interface for the CLEVR ground-truth demo."""

from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
import random
from urllib.parse import parse_qs, urlencode, urlsplit

from pyttr.categorical import Rec, to_latex

from .clevr import AmbiguousReference, CLEVR, CLEVRModel, UnsupportedProgram, clevr_to_h_scene


class CLEVRWebDemo:
    """Prepare scene, question, and TTR data for the web page."""

    def __init__(self, data_dir, split="val"):
        self.dataset = CLEVR(data_dir, split=split)
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

    def evaluate(self, scene, question, answer_requested=False):
        model = CLEVRModel(clevr_to_h_scene(scene))
        try:
            question_record = model.clevr_to_question_rec(question)
        except UnsupportedProgram as exc:
            return {"error": str(exc), "answer": None, "take_latex": None,
                    "meaning_type_latex": None, "answer_semantics_latex": None,
                    "semantic_note": None}

        # Keep operational metadata such as unique_checks and branches outside
        # the displayed semantic record.
        semantic_record = Rec({
            label: question_record[label]
            for label in ("bg", "intrp", "clfr")
            if label in question_record
        })
        result = {
            "error": None,
            "question_latex": semantic_record.to_latex(vars=[]),
            "meaning_type_latex": (
                question_record["meaning_type"].to_latex(vars=[])
                if "meaning_type" in question_record else None
            ),
            "semantic_note": question_record.get("semantic_note"),
            "answer": None,
            "take_latex": None,
            "answer_semantics_latex": None,
        }
        if not answer_requested:
            return result
        try:
            answer, take = model.answer(question)
        except (UnsupportedProgram, AmbiguousReference) as exc:
            result["error"] = str(exc)
            return result
        result["answer"] = answer
        result["take_latex"] = to_latex(self._compact_take(model, question_record, take), vars=[])
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
    def _compact_take(model, question_record, take):
        """Retain individual assignments and omit repetitive evidence payloads."""
        if "branches" in question_record:
            labels = list(dict.fromkeys(
                label
                for branch in question_record["branches"]
                for label in model.individual_labels(branch["background"])
            ))
        else:
            labels = model.individual_labels(question_record["bg"])

        def compact(record):
            return Rec({label: record[label] for label in labels if label in record})

        return [compact(record) for record in take] if isinstance(take, list) else compact(take)

    def image_path(self, scene_index):
        scene = self.dataset.scenes[scene_index]
        path = self.dataset.get_image_path(scene).resolve()
        root = self.dataset.clevr_dir.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(path)
        return path

    def render(self, scene_index=None, question_index=None, answer_requested=False):
        scene, questions, question = self.select(scene_index, question_index)
        result = self.evaluate(scene, question, answer_requested)
        selected_index = question["question_index"]
        query = urlencode({"scene": scene["image_index"], "question": selected_index})
        options = "\n".join(
            f'<option value="{item["question_index"]}"'
            f'{" selected" if item is question else ""}>'
            f'{escape(str(item["question_index"]))} · {escape(item["question"])}</option>'
            for item in questions
        )
        overlays = "\n".join(
            f'<span class="object-id" data-x="{float(obj["pixel_coords"][0])}" '
            f'data-y="{float(obj["pixel_coords"][1])}">{scene["image_index"]}-{index}</span>'
            for index, obj in enumerate(scene["objects"])
            if len(obj.get("pixel_coords", ())) >= 2
        )
        question_panel = (
            f'<div class="math scroll">\\[{escape(result["question_latex"])}\\]</div>'
            if not result["error"] else f'<p class="notice">{escape(result["error"])}</p>'
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
                f'{answer_semantics}'
                '<div class="answer-row">'
                f'<span><small>Answer</small>{escape(str(result["answer"]))}</span>'
                f'<span><small>Dataset</small>{escape(str(question.get("answer", "—")))}</span>'
                '</div>'
            )
        disabled = " disabled" if result["error"] else ""
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
      <div><span class="eyebrow">SPINLS</span><h1>CLEVR ground-truth demo</h1></div>
      <div class="scene-actions">
        <form class="scene-picker" method="get" action="/">
          <label for="scene">Image</label>
          <input id="scene" name="scene" type="number" min="0" step="1" value="{scene["image_index"]}" aria-label="CLEVR image index">
          <button type="submit">Go</button>
        </form>
        <a class="quiet-button" href="/">Random scene</a>
      </div>
    </header>
    <section class="workspace">
      <article class="card scene-card">
        <div class="card-heading"><h2>Scene {scene["image_index"]}</h2><label><input id="show-ids" type="checkbox"> Object IDs</label></div>
        <div class="scene-canvas">
          <img id="scene-image" src="/image?scene={scene["image_index"]}" alt="CLEVR scene {scene["image_index"]}">
          <div id="object-labels" hidden>{overlays}</div>
        </div>
      </article>
      <article class="card question-card">
        <form method="get" action="/">
          <input type="hidden" name="scene" value="{scene["image_index"]}">
          <label class="field-label" for="question">Question</label>
          <select id="question" name="question" onchange="this.form.submit()">{options}</select>
        </form>
        <p class="question-text">{escape(question["question"])}</p>
        <h2>TTR representation</h2>
        {meaning_type_panel}
        {question_panel}
        {semantic_note}
        <form method="post" action="/answer?{query}">
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
h3 { margin:16px 0 8px; color:var(--muted); font-size:12px; letter-spacing:.08em; text-transform:uppercase; }
.eyebrow,.field-label,small { display:block; color:var(--muted); font-size:11px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.workspace { display:grid; grid-template-columns:minmax(0,1.1fr) minmax(340px,.9fr); gap:18px; align-items:start; }
.card { border:1px solid var(--line); border-radius:12px; background:var(--paper); box-shadow:0 8px 30px rgba(30,45,37,.045); padding:18px; }
.scene-canvas { position:relative; overflow:hidden; border-radius:8px; background:#e8ebe7; line-height:0; }
.scene-canvas img { display:block; width:100%; height:auto; }
#object-labels { position:absolute; inset:0; }
.object-id { position:absolute; transform:translate(-50%,-50%); padding:3px 6px; border:1px solid rgba(255,255,255,.75); border-radius:4px; background:rgba(18,25,22,.82); color:white; font:600 11px/1.2 ui-monospace,monospace; box-shadow:0 1px 5px #0005; }
label { color:var(--muted); font-size:13px; } input[type=checkbox] { accent-color:var(--accent); vertical-align:-1px; }
select { width:100%; margin:6px 0 14px; border:1px solid var(--line); border-radius:7px; background:white; padding:10px 34px 10px 10px; color:var(--ink); font:inherit; }
.question-text { min-height:46px; margin-bottom:22px; font-size:17px; line-height:1.45; }
.math { min-height:150px; padding:14px; border:1px solid var(--line); border-radius:8px; background:#fbfcfa; }
.compact-math { min-height:0; }
.scroll { overflow:auto; }
button,.quiet-button { display:inline-block; border:1px solid var(--accent); border-radius:7px; padding:9px 14px; background:var(--accent); color:white; font:600 13px/1.2 inherit; text-decoration:none; cursor:pointer; }
.quiet-button { background:transparent; color:var(--accent); }
button { margin-top:16px; } button:disabled { cursor:not-allowed; opacity:.45; }
.evidence-card { margin-top:18px; }
.evidence-card .math { min-height:90px; }
.answer-row { justify-content:flex-start; margin-top:14px; gap:42px; font-size:24px; font-weight:650; }
.answer-row small { margin-bottom:2px; }
.empty,.notice { margin:0; color:var(--muted); }
.notice { padding:12px; border-radius:7px; background:#fff4e8; color:#85501b; }
.semantic-note { margin-top:12px; }
@media (max-width:800px) { main { margin:20px auto; } .workspace { grid-template-columns:1fr; } header { align-items:flex-start; flex-direction:column; } .scene-actions { flex-wrap:wrap; } }
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
            return scene, question

        def _write(self, status, content, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            route = urlsplit(self.path).path
            try:
                scene_index, question_index = self._parameters()
                if route == "/":
                    body = demo.render(scene_index, question_index).encode()
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
                scene_index, question_index = self._parameters()
                body = demo.render(scene_index, question_index, answer_requested=True).encode()
                self._write(200, body, "text/html; charset=utf-8")
            except KeyError as exc:
                self.send_error(404, str(exc))

    return DemoHandler


def serve(data_dir, split="val", host="127.0.0.1", port=8000):
    demo = CLEVRWebDemo(Path(data_dir), split=split)
    server = ThreadingHTTPServer((host, port), make_handler(demo))
    print(f"CLEVR demo running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
