from ttr_clevr import preds, Ind, \
                      scene_graph_to_h_data_db, \
                      find_witness_takes
from pyttr.ttrtypes import PType, Fun, RecType
from pyttr.utils import show, show_latex, print_latex

import json
import pprint
from pathlib import Path

class FormatPrinter(pprint.PrettyPrinter):

    def __init__(self, formats):
        super(FormatPrinter, self).__init__()
        self.formats = formats

    def format(self, obj, ctx, maxlvl, lvl):
        if type(obj) in self.formats:
            return self.formats[type(obj)] % obj, 1, 0
        return pprint.PrettyPrinter.format(self, obj, ctx, maxlvl, lvl)

fprinter = FormatPrinter({float: "%.3f"})

clevr_dir = Path("./data/CLEVR_v1.0")
questions_path = clevr_dir/"questions"/"CLEVR_val_questions.json"
scenes_path = clevr_dir/"scenes"/"CLEVR_val_scenes.json"

with questions_path.open() as f:
    questions = json.load(f)['questions']

with scenes_path.open() as f:
    scenes = json.load(f)['scenes']

def to_image_path(r):
    return clevr_dir/'images'/r['split']/r['image_filename']

q = questions[1156]
g = scenes[q['image_index']]
image_path = to_image_path(q)

T = RecType({
    'x': Ind,
    'x_size': (Fun('v', Ind, PType(preds['small'], ['v'])), ['x']),
    'x_color': (Fun('v', Ind, PType(preds['yellow'], ['v'])), ['x']),
    'x_shape': (Fun('v', Ind, PType(preds['cylinder'], ['v'])), ['x']),
    'y': Ind,
    'y_material': (Fun('v', Ind, PType(preds['metal'], ['v'])), ['y']),
    'c':(Fun('v1',Ind, Fun('v2',Ind, PType(preds['left'], ['v1','v2']))), ['x','y'])
})

h_data_db = scene_graph_to_h_data_db(g)
r = list(find_witness_takes(T, h_data_db, label_map={}))[0]

tex = fr"""
\documentclass{{minimal}}
\usepackage{{amsmath}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{listings}}

\input{{json_listing.tex}}

\begin{{document}}
\includegraphics[width=0.5\textwidth]{{{str(image_path)}}}
\\
{q['question']}

\begin{{lstlisting}}[language=json]
{fprinter.pformat(g)}
\end{{lstlisting}}

\begin{{lstlisting}}[language=json]
{fprinter.pformat(q)}
\end{{lstlisting}}

Type associated with question:

$$
{T.to_latex(vars=[])}
$$

Witness record (extracted from scene graph):

$$
{r.to_latex(vars=[])}
$$


\end{{document}}

"""

with open("clevr.tex", 'w') as f:
    f.write(tex)
