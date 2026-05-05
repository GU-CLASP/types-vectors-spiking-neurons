from util import load_clevr
from pyttr.utils import show, show_latex, print_latex
from pathlib import Path
from pyttr.ttrtypes import LazyObj


from ttr_clevr import *


clevr_dir = Path("./data/CLEVR_v1.0")
questions, scenes = load_clevr(clevr_dir, split='val')

assert all([i == s['image_index'] for i,s in enumerate(scenes)])

is_color_question = lambda q: q['program'][-1]['function'] == 'query_color'
includes_same = lambda q:any(f['function'].startswith('same') for f in q['program'])
includes_intersect = lambda q:any(f['function'] == 'intersect' for f in q['program'])
color_questions = list(filter(is_color_question, questions))
color_questions = list(filter(lambda x: not includes_same(x), color_questions))
color_questions = list(filter(lambda x: not includes_intersect(x), color_questions))
color_questions = color_questions[:100]

# q = questions[1156]
for q in color_questions:
    s = scenes[q['image_index']]
    print(f"{q['question_index']}: {q['question']}")
    print(q['answer'])
    q_rec = clevr_to_question_rec(q)
    h_scene = clevr_to_h_scene(s)
    takes = list(find_witness_takes(q_rec.bg, h_scene))
    assert len(takes) == 1
    s = takes[0]
    answer = q_rec.clfr.app(s).comps.pred.name
    print(answer)

