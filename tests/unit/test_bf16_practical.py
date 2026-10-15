"""Offline BF16-only checkpoint checks; no other model format is executed."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from src.paper_bf16_practical import assert_bf16, restore_whole, save_whole, validate_config
from src.paper_precision import PrecisionLinear


class ToyConfig:
    def to_dict(self):
        return {"model_type": "toy"}


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ToyConfig()
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)])

    def forward(self, x):
        return self.layers[0](x)


class BF16PracticalTests(unittest.TestCase):
    def test_checkpoint_preserves_bf16_factors_and_output(self):
        model = Toy().eval()
        model.layers[0] = PrecisionLinear(torch.eye(4, 2, dtype=torch.bfloat16),
                                         torch.ones(2, 4, dtype=torch.bfloat16))
        values = torch.ones(1, 4, dtype=torch.bfloat16)
        before = model(values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"
            info = save_whole(model, path)
            restored = restore_whole(path, device="cpu", model_factory=lambda _: Toy())
            assert_bf16(restored)
            self.assertTrue(torch.equal(before, restored(values)))
            self.assertEqual(restored.layers[0].left.dtype, torch.bfloat16)
            self.assertEqual(restored.layers[0].right.dtype, torch.bfloat16)
            self.assertGreater(info["bytes"], 0)
            with self.assertRaises(FileExistsError):
                save_whole(model, path)

    def test_dense_checkpoint_reload_uses_bf16(self):
        model = Toy().eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"dense.pt"
            save_whole(model, path)
            restored = restore_whole(path, device="cpu", model_factory=lambda _: Toy())
            self.assertEqual(restored.layers[0].weight.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(model.layers[0].weight, restored.layers[0].weight))

    def test_current_config_is_two_seed_and_sidecar_validated(self):
        config = json.loads(Path("configs/paper_bf16_practical_two_seeds_20261014.json").read_text())
        validate_config(config)
        self.assertEqual(config["seeds"], [11, 23])
        for replacement in ([11, 23, 37], [11, 11], [23, 37], []):
            with self.assertRaises(ValueError):
                validate_config({**config, "seeds": replacement})
        with self.assertRaises(ValueError):
            validate_config({**config, "factor_formats": ["bfloat16", "int8"]})


if __name__ == "__main__":
    unittest.main()
