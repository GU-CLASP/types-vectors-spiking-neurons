import json
import pprint
from pathlib import Path

from pyttr.ttrtypes import Type, BType, Pred, PType, RecType, Fun
from pyttr.utils import show, show_latex, print_latex

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
s = scenes[q['image_index']]
image_path = to_image_path(q)

################
Ind = BType('Ind')
hug = Pred('hug',[Ind,Ind])

boy = Pred('boy',[Ind])
dog = Pred('dog',[Ind])

T = RecType({
        'x':Ind,
        'c_boy':(Fun('v',Ind,PType(boy,['v'])), ['x']),
        'y':Ind,
        'c_dog':(Fun('v',Ind,PType(dog,['v'])), ['y']),
        'e':(Fun('v1',Ind,Fun('v2',Ind, PType(hug,['v1','v2']))), ['x','y'])
    })
################


from IPython import embed; embed(colors="neutral"); raise;

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
{fprinter.pformat(s)}
\end{{lstlisting}}

\begin{{lstlisting}}[language=json]
{fprinter.pformat(q)}
\end{{lstlisting}}

$$
{T.to_latex(vars=[])}
$$



\end{{document}}

"""

with open("clevr.tex", 'w') as f:
    f.write(tex)

