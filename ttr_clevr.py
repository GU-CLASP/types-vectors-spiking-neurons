from pyttr.ttrtypes import Type, BType, BTypeClass, Pred, PType, RecType, Fun, TypeClass
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
