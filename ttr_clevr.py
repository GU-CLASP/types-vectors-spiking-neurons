import json
import pprint
from pathlib import Path

import string

from pyttr.ttrtypes import Type, BType, BTypeClass, Pred, PType, RecType, Fun, TypeClass
from pyttr.utils import show, show_latex, print_latex
from pyttr.records import Rec

from itertools import permutations

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

################

funcs = [action['value_inputs'][0] for q in questions for action in q['program'] if action['function'] == 'filter_shape']

list(set(funcs))

Ind = BType('Ind')

clevr_attributes = {
        'color': ['purple', 'brown', 'gray', 'yellow', 
                  'green', 'blue', 'cyan', 'red'],
        'size': ['large', 'small'],
        'material': ['metal', 'rubber'],
        'shape': ['sphere', 'cube', 'cylinder']
    }

clevr_relations = ['right', 'behind', 'front', 'left']

preds = {}
for key in clevr_attributes:
    for value in clevr_attributes[key]:
        # The witness_funs of a Pred must return a type (given arguments)
        # this is somewhat of a hack but we use BTypes to back up PTypes since
        # BTypes allow for winness conditions 
        pred = Pred(value, [Ind])
        btype = BType(f"{value}")
        btype.learn_witness_condition(
            lambda args, key=key, value=value: args[0][key] == value)
        pred.learn_witness_fun(
            lambda args, btype=btype: btype)
        preds[value] = pred
for key in clevr_relations:
    pred = Pred(key, [Ind, Ind])
    btype = BType(f"{key}")
    btype.learn_witness_condition(
            lambda args, key=key: args[1]['id'] in args[0][key])
    pred.learn_witness_fun(
            lambda args, btype=btype: btype)
    preds[key] = pred

################

def scene_graph_to_h_data_db(g):
    h_data_db = {}
    for i, o in enumerate(g['objects']):
        h_data = o.copy()
        h_data['id'] = f"{g['image_index']}-{i}"
        for rel, values in g['relationships'].items():
            h_data[rel] = [f"{g['image_index']}-{j}" for j in values[i]]
        h_data_db[h_data['id']] = h_data
    return h_data_db 

def get_ind_labels(T : RecType):
    return [l for l, T_ in T.comps.__dict__.items() if T_ == Ind]

def clevr_question_function_to_record_type(q):
    pass

def get_restricted_type(btype_labels, rectype):

    type_labels = get_ind_labels(rectype)
    res = RecType({})

    for label in type_labels:
        T = rectype.comps.__getattribute__(label)
        if isinstance(T, BTypeClass):
            if label in btype_labels:
                res.addfield(label, T)
        elif isinstance(T, tuple): # dependent type
            fun, args = T
            if all(label_ in btype_labels for label_ in args):
                res.addfield(label, T)
        elif isinstance(T, RecType):
            T_ = get_restricted_type(labels, T)
            res.addfield(label, T_)
            # TODO: what happens if it's the empty record type? is that ok?
        else:
            raise NotImplementedError(
                f"Don't know how to restrict values of type\
                        {type(field_type)}."
                )

    return res

def h_data_to_sit_take(sit_type, label_map, h_data_db):
    """
    Gives a take on the situation (a Record) based on h_data_db. 
    The take is guided by a situation type and the mapping of 
    labels to individuals is goverened by label_map. 

    We'll assume that the labels of all Ind fields in sit_type 
    appear in label_map.
    """

    res = Rec({}) 
    for label, T in sit_type.comps.__dict__.items():
        if isinstance(T, BTypeClass):
            h_data_id = label_map[label]
            res.addfield(label, h_data_id)
            T.judge(h_data_id) # judge the entity to be an Ind
        elif isinstance(T, tuple):
            fun, args = T
            rec_args = tuple(h_data_db[label_map[l_]] for l_ in args)
            res.addfield(label, rec_args)
        elif isinstance(T, RecType):
            res.add_field(h_data_to_sit_take(T, label_map, h_data_db))
        else:
            raise NotImplementedError(
                f"Don't know how to have a take on values of type\
                        {type(field_type)}."
                )
    return res 


def find_witness_takes(sit_type, h_data_db, label_map={}):

    type_labels = get_ind_labels(sit_type)
    assigned_labels = set(label_map.keys())
    unassigned_labels = {l for l in type_labels if not l in assigned_labels}

    # stopping condition: we have no more labels to assign.
    # if the (complete) record is proof of the type, yield it.
    if not unassigned_labels:
        sit_take = h_data_to_sit_take(sit_type, label_map, h_data_db)
        if sit_type.query(sit_take):
            yield sit_take

    # recursive condition: the lable_map in incomplete
    # check if the sit_take is consistent with the type restricted to
    # the incomplete label map. if so we want to try extensions of it.
    else:
        sit_type_restr = get_restricted_type(assigned_labels, sit_type)
        sit_take = h_data_to_sit_take(sit_type_restr, label_map, h_data_db)
        if sit_type_restr.query(sit_take):
            l = unassigned_labels.pop()
            for h_id in h_data_db:
                label_map_extended = label_map | {l: h_id}
                yield from find_witness_takes(
                        sit_type, h_data_db, label_map_extended
                    )


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

$$
{T.to_latex(vars=[])}
$$

$$
{r.to_latex(vars=[])}
$$


\end{{document}}

"""

with open("clevr.tex", 'w') as f:
    f.write(tex)
