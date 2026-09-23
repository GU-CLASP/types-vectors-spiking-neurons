import json
import pprint

class CLEVR:
    def __init__(self, clevr_dir, split='train'):

        self.clevr_dir = clevr_dir
        qp = self.clevr_dir/"questions"/f"CLEVR_{split}_questions.json"
        sp = self.clevr_dir/"scenes"/f"CLEVR_{split}_scenes.json"

        with qp.open() as f:
            self.questions = json.load(f)['questions']

        with sp.open() as f:
            self.scenes = json.load(f)['scenes']

    def get_image_path(self, r):
        return self.clevr_dir/'images'/r['split']/r['image_filename']

class FormatPrinter(pprint.PrettyPrinter):

    def __init__(self, formats):
        super(FormatPrinter, self).__init__()
        self.formats = formats

    def format(self, obj, ctx, maxlvl, lvl):
        if type(obj) in self.formats:
            return self.formats[type(obj)] % obj, 1, 0
        return pprint.PrettyPrinter.format(self, obj, ctx, maxlvl, lvl)

fprinter = FormatPrinter({float: "%.3f"})



