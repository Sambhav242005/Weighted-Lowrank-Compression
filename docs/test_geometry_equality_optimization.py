"""CPU parity checks for the pending exact-equality optimization.

Expected to pass before and after the repair; the branch-avoidance check
requires the repair to be applied, or an isolated in-memory patched class.
"""
import math
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from src.paper_comparisons import Geometry


class RepairedG4Geometry:
    """Frozen generic precision-repaired accumulator, without equality shortcut."""
    def __init__(self):
        self.error = self.reference = self.candidate = self.cosine = self.cka = 0.0
        self.tokens = self.valid_cka_tokens = 0

    def add(self, reference, candidate):
        x = reference.float().reshape(-1, reference.shape[-1])
        y = candidate.float().reshape(-1, candidate.shape[-1])
        if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise FloatingPointError('invalid diagnostic activations')
        self.error += (x-y).double().square().sum().item()
        self.reference += x.double().square().sum().item()
        self.candidate += y.double().square().sum().item()
        self.cosine += (1-F.cosine_similarity(x, y, dim=-1)).double().sum().item()
        self.tokens += len(x)
        if len(x) > 1:
            x, y = x-x.mean(0), y-y.mean(0)
            scale_x, scale_y = x.abs().amax(), y.abs().amax()
            if scale_x.item() > 0 and scale_y.item() > 0:
                x, y = x/scale_x, y/scale_y
                gx, gy = (x @ x.T).double(), (y @ y.T).double()
                norm_x, norm_y = gx.square().sum().sqrt(), gy.square().sum().sqrt()
                if norm_x.item() > 0 and norm_y.item() > 0:
                    similarity = (gx*gy).sum()/(norm_x*norm_y)
                    if not torch.isfinite(similarity):
                        raise FloatingPointError('nonfinite linear CKA')
                    self.cka += similarity.clamp(0, 1).item()*len(x)
                    self.valid_cka_tokens += len(x)

    def result(self):
        return {'normalized_squared_error': self.error/max(self.reference, 1e-30),
                'cosine_distance': self.cosine/max(self.tokens, 1),
                'norm_ratio': math.sqrt(self.candidate/max(self.reference, 1e-30)),
                'window_linear_cka': self.cka/self.valid_cka_tokens if self.valid_cka_tokens else None,
                'valid_cka_tokens': self.valid_cka_tokens, 'tokens': self.tokens}


class GeometryEqualityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def assertParity(self, batches):
        reference, optimized = RepairedG4Geometry(), Geometry()
        for x, y in batches:
            reference.add(x, y)
            optimized.add(x, y)
        old, new = reference.result(), optimized.result()
        self.assertEqual(set(old), set(new))
        for key in old:
            with self.subTest(metric=key):
                if old[key] is None:
                    self.assertIsNone(new[key])
                elif key in {'tokens', 'valid_cka_tokens'}:
                    self.assertEqual(old[key], new[key])
                else:
                    self.assertAlmostEqual(old[key], new[key], delta=1e-12)

    def test_identical_inputs_at_all_scales(self):
        data = torch.randn(64, 128)
        for scale in (0.0, 1e-10, 1.0, 1e4):
            x = data*scale
            with self.subTest(scale=scale):
                self.assertParity([(x, x.clone())])

    def test_constant_and_singleton_identity_covariance_remains_undefined(self):
        for x in (torch.zeros(7, 9), torch.ones(7, 9)*1e4, torch.randn(1, 9)):
            self.assertParity([(x, x.clone())])
            optimized = Geometry()
            optimized.add(x, x.clone())
            self.assertIsNone(optimized.result()['window_linear_cka'])

    def test_zero_and_tiny_norm_cosine_epsilon_is_preserved(self):
        for x in (torch.zeros(4, 3), torch.ones(4, 3)*1e-10):
            self.assertParity([(x, x.clone())])
            optimized = Geometry()
            optimized.add(x, x.clone())
            self.assertGreater(optimized.result()['cosine_distance'], .99)

    def test_mixed_identical_and_nonidentical_windows_accumulate(self):
        a, b = torch.randn(7, 9), torch.randn(11, 9)
        self.assertParity([(a*1e4, a.clone()*1e4), (b, b+.1),
                           (torch.ones(4, 9), torch.ones(4, 9)),
                           (a*1e-10, a.clone()*1e-10), (a, b[:7])])

    def test_equal_branch_never_calls_gram_matrix_multiplication(self):
        x = torch.randn(7, 9)
        with mock.patch.object(torch.Tensor, '__matmul__', side_effect=AssertionError('Gram matmul executed')):
            optimized = Geometry()
            optimized.add(x, x.clone())
        self.assertEqual(optimized.result()['window_linear_cka'], 1.0)

    def test_nonidentical_branch_does_not_skip_gram_computation(self):
        x = torch.randn(7, 9)
        with mock.patch.object(torch.Tensor, '__matmul__', side_effect=AssertionError('expected generic Gram path')):
            with self.assertRaisesRegex(AssertionError, 'expected generic Gram path'):
                Geometry().add(x, x+.01)

    def test_equal_nonfinite_inputs_are_still_rejected(self):
        for value in (float('nan'), float('inf')):
            x = torch.full((7, 9), value)
            with self.assertRaises(FloatingPointError):
                Geometry().add(x, x)


if __name__ == '__main__':
    unittest.main()
