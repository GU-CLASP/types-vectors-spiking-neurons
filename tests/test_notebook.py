"""Offline checks of retained notebook syntax, HD helpers, and the new TTR bridge."""
import ast
import json
from pathlib import Path
import unittest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "ttr-hdc-glue_cifar10.ipynb"


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text())

    def test_all_code_cells_parse(self):
        for i, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                with self.subTest(cell=i):
                    ast.parse("".join(cell["source"]))
                    self.assertEqual(cell["outputs"], [])

    def test_hd_helpers_and_ttr_bridge_with_synthetic_vectors(self):
        try:
            import torch
            import torchhd
        except ImportError:
            self.skipTest("Optional torch/torchhd dependencies not installed")
        namespace = {"torch": torch, "torchhd": torchhd, "F": torch.nn.functional}
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                           and node.name in {"concensus_sum", "compact_concensus_sum", "HDGlue"}]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)

        majority = namespace["concensus_sum"](torch.tensor([[[True, False], [True, False], [False, True]]]))
        self.assertEqual(majority.tolist(), [[True, False]])
        memory = torch.tensor([[3, 0], [0, 2]])
        totals = torch.tensor([3, 2])
        before = memory.clone(), totals.clone()
        namespace["compact_concensus_sum"](memory, totals)
        self.assertTrue(torch.equal(before[0], memory))
        self.assertTrue(torch.equal(before[1], totals))
        glue = namespace["HDGlue"](4, 16, 10)
        self.assertEqual(tuple(glue(torch.zeros(1, 4)).shape), (1, 16))
        self.assertEqual(tuple(glue(torch.full((1, 4), 1e6)).shape), (1, 16))

        namespace.update(M=torch.tensor([False, False, False, False]),
                         class_ids=torch.tensor([[False] * 4, [True] * 4]), labels=["cube", "sphere"],
                         eg_image_dense=torch.zeros(1, 4), hd_glue=lambda x: torch.zeros(1, 4, dtype=torch.bool))
        bridge = next(c for c in self.notebook["cells"] if "ttr-bridge" in c["metadata"].get("tags", []))
        exec(compile("".join(bridge["source"]), str(NOTEBOOK), "exec"), namespace)
        self.assertEqual(namespace["judgements"], {"cube": True, "sphere": False})
        types = namespace["class_types"]
        self.assertIs(types["sphere"].query((True,) * 4), True)
        namespace["M"].fill_(True)
        self.assertIs(types["cube"].query((False,) * 4), True)


if __name__ == "__main__":
    unittest.main()
