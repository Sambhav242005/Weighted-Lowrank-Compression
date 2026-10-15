"""CPU contract checks; execute with python -m unittest discover."""
import tempfile
import unittest
from pathlib import Path

import torch

from src.paper_comparisons import Geometry, fit_checkpoint, select_result, student_from_checkpoint, validate_selection_provenance
from src.paper_math import weighted_factors
from src.compression_core import fit_activation_weighted_low_rank


class ToyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(5, 4).half()


class ComparisonTests(unittest.TestCase):
    def test_frozen_selection_rejects_changed_teacher_splits_tokens_and_recipes(self):
        import copy
        config = {'model_revision': 'model-sha', 'dataset_revision': 'data-sha', 'rank': 2,
                  'precision': 'float16', 'seeds': [11, 23, 37]}
        manifests = {s: {'hash': s+'-sha'} for s in ['train', 'validation', 'test']}
        manifests['calibration_hashes'] = {str(s): f'calib-{s}' for s in config['seeds']}
        targets = [{'name': 'self_attn.o_proj', 'rank': 2, 'shape': [4, 5], 'factor_parameters': 18}]
        selection = {**config, 'state': 'frozen', 'selection_split': 'validation', 'selection_seed': 11,
                     'teacher_hash': 'teacher-sha', 'dataset_content_hash': 'dataset-bytes-sha',
                     'split_hashes': {s: manifests[s]['hash'] for s in ['train', 'validation', 'test']},
                     'calibration_token_hashes': manifests['calibration_hashes'], 'targets': targets}
        validate_selection_provenance(selection, config, manifests, 'teacher-sha', targets, 'dataset-bytes-sha')
        for key, corrupted in [('teacher_hash', 'changed'), ('split_hashes', {'test': 'changed'}),
                               ('calibration_token_hashes', {'11': 'changed'}), ('targets', []),
                               ('factor_convention', 'balanced'), ('model_revision', 'changed'),
                               ('dataset_content_hash', 'changed')]:
            mutated = copy.deepcopy(selection)
            mutated[key] = corrupted
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_selection_provenance(mutated, config, manifests, 'teacher-sha', targets, 'dataset-bytes-sha')

    def test_measured_biased_output_matches_recorded_solver(self):
        torch.manual_seed(11)
        w, x, bias = torch.randn(4, 5), torch.randn(5, 60), torch.randn(4)
        y = w @ x + bias[:, None] + 0.02*torch.randn(4, 60)
        for rank in [2, 4]:
            left, right = weighted_factors(w, x, rank, beta=0.1, ridge_fraction=0.01,
                                           outputs=y, bias=bias)
            recorded = fit_activation_weighted_low_rank(w, x, y, rank, beta=0.1,
                                                       ridge_fraction=0.01, bias=bias)
            torch.testing.assert_close(left @ right, recorded, rtol=3e-5, atol=3e-5)
        with self.assertRaises(ValueError):
            weighted_factors(w, x, 2, outputs=y[:, :-1], bias=bias)
        with self.assertRaises(ValueError):
            weighted_factors(w, x, 2, outputs=torch.full_like(y, float('nan')))

    def test_geometry_accumulates_not_last_batch(self):
        geometry = Geometry()
        geometry.add(torch.ones(2, 3), torch.zeros(2, 3))
        geometry.add(torch.ones(2, 3), torch.ones(2, 3))
        result = geometry.result()
        self.assertEqual(result['tokens'], 4)
        self.assertAlmostEqual(result['normalized_squared_error'], 0.5)
        self.assertAlmostEqual(result['norm_ratio'], 2**-0.5)

    def test_selection_ties_use_less_shrinkage_then_anchor(self):
        records = [{'id': str(i), 'specification': {'method': 'anchored', 'beta': beta,
                   'ridge_fraction': ridge}, 'metrics': {'nll': nll}}
                   for i, (beta, ridge, nll) in enumerate([(0.1, 0.01, 2.0), (0.3, 0, 2.0), (0.1, 0, 2.0000005)])]
        self.assertEqual(select_result(records, {'anchored'})['id'], '2')

    def test_checkpoint_roundtrip_preserves_teacher_and_parameter_budget(self):
        teacher = ToyTeacher()
        before = {n: p.clone() for n, p in teacher.state_dict().items()}
        with tempfile.TemporaryDirectory() as root:
            checkpoint, metadata = fit_checkpoint(teacher, {}, {'rank': 2, 'pseudoinverse_rcond': 1e-7},
                                                  {'method': 'svd'}, Path(root)/'run')
            student = student_from_checkpoint(teacher, checkpoint)
            self.assertEqual(metadata['factor_parameters'], 2*(5+4))
            self.assertGreater(metadata['serialized_bytes'], 0)
            self.assertFalse(hasattr(student.self_attn.o_proj, 'weight'))
            x = torch.randn(3, 5).half()
            state = torch.load(checkpoint, weights_only=True)['self_attn.o_proj']
            expected = (x @ state['v'].T) @ state['u'].T + state['bias']
            torch.testing.assert_close(student.self_attn.o_proj(x), expected)
            for n, p in teacher.state_dict().items():
                torch.testing.assert_close(p, before[n], rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                fit_checkpoint(teacher, {}, {'rank': 2}, {'method': 'svd'}, Path(root)/'run')


if __name__ == '__main__':
    unittest.main()
