from pyttr.ttrtypes import Type, BType, BTypeClass, Pred, PType, RecType, Fun, TypeClass, FunType
from pyttr.records import Rec

Ind = BType('Ind')

clevr_attributes = {
        'color': ['purple', 'brown', 'gray', 'yellow', 
                  'green', 'blue', 'cyan', 'red'],
        'size': ['large', 'small'],
        'material': ['metal', 'rubber'],
        'shape': ['sphere', 'cube', 'cylinder']
    }
clevr_relations = ['right', 'behind', 'front', 'left']

variables = ['x', 'y', 'z'] + ['x{i}' for i in range(1, 10)]

# define "classifiers" based on ground truth metadata
classif_gt = {}
for attr in clevr_attributes:
    for value in clevr_attributes[attr]:
        classif_gt[value] = lambda args, a=attr, v=value: args[0][a] == v 
for rel in clevr_relations:
    classif_gt[rel] = lambda args, r=rel: args[0]['id'] in args[1][r]

# define predicates corresponding to each of the classifiers
# this is somewhat of a hack (?) but we use BTypes 
# to back up PTypes since BTypes can have witness conditions 
# the witness_funs of a Pred must return a type (given arguments)
preds = {}
for attr in clevr_attributes:
    for value in clevr_attributes[attr]:
        pred = Pred(value, [Ind])
        btype = BType(f"{value}")
        btype.learn_witness_condition(classif_gt[value])
        pred.learn_witness_fun(lambda args, btype=btype: btype)
        preds[value] = pred
for rel in clevr_relations:
    pred = Pred(rel, [Ind, Ind])
    btype = BType(f"{rel}")
    btype.learn_witness_condition(classif_gt[rel])
    pred.learn_witness_fun(lambda args, btype=btype: btype)
    preds[rel] = pred


# ans_types = {}
# for attr in clevr_attributes:
    # ans_types[attr] = Fun('v', Ind, PType(preds[

class AttrLookup:
    def __init__(self, attr, var):
        self.var = var
        self.label = f'{var}_{attr}'
        self.vals = clevr_attributes[attr]
    def subst(self, var, arg):
        h_data = arg.__getattribute__(self.label)
        for val in self.vals:
            if classif_gt[val](h_data):
                # return RecType({ self.label: PType(preds[val], self.var) })
                return PType(preds[val], self.var) 
        raise ValueError(f"Querying {arg} for attribute: {self.label} but it does not witness any of its respective PTypes")

def clevr_to_h_scene(g):
    h_scene = {}
    for i, o in enumerate(g['objects']):
        h_data = o.copy()
        h_data['id'] = f"{g['image_index']}-{i}"
        for rel, values in g['relationships'].items():
            h_data[rel] = [f"{g['image_index']}-{j}" for j in values[i]]
        h_scene[h_data['id']] = h_data
    return h_scene 


def clevr_to_question_rec(q):
    """
    Takes a "ground truth" representation of a question in the CLEVR
    funcitonal program format and conversts it to a question record.
    In the Larsson (2024) style.
    """
    _variables = list(reversed(variables.copy()))
    func_comps = ('inputs', 'function', 'value_inputs')
    bg = { }
    for func in q['program']:
        input_idxs, f, vals = map(lambda x: func[x], func_comps)
        if f == 'scene':
            v = _variables.pop()
            bg[v] = Ind
        elif f == 'relate':
            assert len(vals) == 1
            val = vals[0]
            u = _variables.pop()
            bg[u] = Ind
            bg[f'{u}{v}_{val}'] = (Fun('u', Ind, 
                                       Fun('v', Ind, 
                                           PType(preds[val], ['u', 'v']))), 
                                   [u, v])
            v = u
        elif f.startswith('filter'):
            _, attr = f.split('_')
            assert len(vals) == 1
            val = vals[0]
            bg[f'{v}_{attr}'] = (Fun('v', Ind, PType(preds[val], ['v'])), [v])
        elif f == 'unique':
            continue
        elif f.startswith('query'):
            _, attr = f.split('_')
            T_bg = RecType(bg)
            # interp = Fun( #TODO
                # 'r', T_bg, 
                # Fun(
                    # 'P', FunType(Ind, Type), 
                    # RecType({ f'c_{attr}-{v}' : PType( })
                # )
            # )
            clfr = Fun(
                'r', T_bg, 
                AttrLookup('color', v)
            )
            q_rec = Rec({
                'bg': T_bg, 
                # 'interp': interp, 
                'clfr': clfr
            })
        else:
            raise ValueError(f"Unknown CLEVR question function type: {f}")
    return q_rec

def get_ind_labels(T : RecType):
    return [l for l, T_ in T.comps.__dict__.items() if T_ == Ind]

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

def h_data_to_sit_take(h_scene, label_map):
    """
    Gives a take on the situation (a Record) based on h_scene. 
    The take is guided by the label_map
    """

    label_map_rev = {h_id: l for l, h_id in label_map.items()}
    res = Rec({}) 
    for l, h_id in label_map.items():
        h_hash = h_scene[h_id]
        res.addfield(l, h_id)
        Ind.judge(h_id)
        for attr in clevr_attributes:
            res.addfield(f'{l}_{attr}', (h_hash,))
        for rel in clevr_relations:
            for h_id_ in h_hash[rel]:
                if not h_id_ in label_map_rev:
                    continue # object related to an object outside the map
                h_hash_ = h_scene[h_id_]
                l_ = label_map_rev[h_id_]
                res.addfield(f'{l_}{l}_{rel}', (h_hash_, h_hash))
    return res

def build_witness_takes(T, h_scene, label_map={}):
    """
    Given a visual scene (in the form of `h_scene`), find takes on
    the scene (list of Rec), if any, that satisfy a given a sit type `T`.

    `label_map` is a (possibly empty) partial map from `T` labels 
    to `h_scene` ids. Any resulting witness takes will repsect `label_map`.
    """

    type_labels = get_ind_labels(T)
    free_type_labels = {l for l in type_labels if not l in label_map.keys()}
    aux_labels = [f'v{i}' for i in range(1,10)]
    assert not any([v in type_labels for v in aux_labels])

    # stopping condition: no more labels to assign
    #   if the (complete) record is proof of the type, yield it.
    if not free_type_labels:
        unmapped_ids = [h_id for h_id in h_scene 
                         if not h_id in label_map.values()]
        label_map |= {l: h_id for h_id, l in zip(unmapped_ids, aux_labels)}
        s = h_data_to_sit_take(h_scene, label_map)
        if T.query(s):
            yield s

    # recursive condition: the lable_map is incomplete
    #   check if the s is consistent with the type (restricted to
    #   the incomplete label map). if so we want to try extensions of it.
    else:
        T_restr = get_restricted_type(label_map.keys(), T)
        s = h_data_to_sit_take(h_scene, label_map)
        if T_restr.query(s):
            l = free_type_labels.pop()
            for h_id in h_scene:
                label_map_extended = label_map | {l: h_id}
                yield from build_witness_takes(
                        T, h_scene, label_map_extended
                    )
