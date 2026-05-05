import json

def load_clevr(clevr_dir, split='train'):
    qp = clevr_dir/"questions"/f"CLEVR_{split}_questions.json"
    sp = clevr_dir/"scenes"/f"CLEVR_{split}_scenes.json"

    with qp.open() as f:
        questions = json.load(f)['questions']

    with sp.open() as f:
        scenes = json.load(f)['scenes']

    return questions, scenes


