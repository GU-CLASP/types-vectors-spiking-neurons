from ttr_clevr import *

from util import CLEVR, fprinter
from pathlib import Path

clevr_dir = Path("./data/CLEVR_v1.0")
data = CLEVR(clevr_dir, split='val')

# q = data.questions[117994]
q = data.questions[115673]

g = data.scenes[q['image_index']]
image_path = data.get_image_path(q)
q_rec = clevr_to_question_rec(q)

h_scene = clevr_to_h_scene(g)
takes = list(build_witness_takes(q_rec.bg, h_scene))
s = takes[0]

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

Record, $q_{{rec}}$ associated with question:

$$
{q_rec.to_latex(vars=[])}
$$

Witness record, $s$ (extracted from scene graph):

$$
{s.to_latex(vars=[])}
$$


Answer:

$$
q_{{rec}}.clfr(s) =  {q_rec.clfr.app(s).comps.pred.to_latex(vars=[])}
$$

\end{{document}}

"""

with open("clevr.tex", 'w') as f:
    f.write(tex)

