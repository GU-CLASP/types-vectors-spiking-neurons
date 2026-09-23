from util import CLEVR
from pyttr.utils import show, show_latex, print_latex
from pathlib import Path

from ttr_clevr import *

clevr_dir = Path("./data/CLEVR_v1.0")
data = CLEVR(clevr_dir, split='val')

is_color_question = lambda q: q['program'][-1]['function'] == 'query_color'
no_same = lambda q: not any(f['function'].startswith('same') 
                            for f in q['program'])
no_intersect = lambda q: not any(f['function'] == 'intersect' for
                            f in q['program'])

selected_questions = list(filter(
    is_color_question, filter(
        no_same, filter(
            no_intersect, data.questions))))[:100]

# questions = [questions[2420]]

for q in selected_questions:
    g = data.scenes[q['image_index']]
    print(f"{q['question_index']}: {q['question']}")
    print(q['answer'])
    q_rec = clevr_to_question_rec(q)
    h_scene = clevr_to_h_scene(g)
    takes = list(build_witness_takes(q_rec.bg, h_scene))
    if not len(takes) == 1:
        print(f"Found {len(takes)} valid takes.")
        continue
    s = takes[0]
    answer = q_rec.clfr.app(s).comps.pred.name
    print(answer)
    print()

