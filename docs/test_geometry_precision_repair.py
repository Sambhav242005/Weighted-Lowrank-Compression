"""Run after the controlled boundary repair; current source intentionally fails.

rtk proxy .venv-cuda/Scripts/python.exe -m unittest discover -s docs -p test_geometry_precision_repair.py -v
"""
import unittest

import torch

from src.paper_comparisons import Geometry


class GeometryPrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_identity_cka_across_small_and_large_scales(self):
        reference = torch.randn(64, 128)
        for scale in (1e-10, 1.0, 1e4):
            with self.subTest(scale=scale):
                geometry = Geometry()
                geometry.add(reference*scale, reference*scale)
                result = geometry.result()
                self.assertAlmostEqual(result['window_linear_cka'], 1.0, places=6)
                self.assertEqual(result['valid_cka_tokens'], 64)
                self.assertAlmostEqual(result['normalized_squared_error'], 0.0)

    def test_independent_scale_invariance_near_half_precision_limit(self):
        reference = torch.randn(64, 128).clamp(-3, 3)*20000
        reference = reference.half()
        self.assertTrue(torch.isfinite(reference).all())
        geometry = Geometry()
        geometry.add(reference, reference.float()*1e-12)
        self.assertAlmostEqual(geometry.result()['window_linear_cka'], 1.0, places=6)

    def test_feature_rotation_preserves_alignment(self):
        reference = torch.randn(64, 16)
        rotation, _ = torch.linalg.qr(torch.randn(16, 16))
        geometry = Geometry()
        geometry.add(reference, reference @ rotation)
        self.assertAlmostEqual(geometry.result()['window_linear_cka'], 1.0, places=6)

    def test_orthogonal_token_subspaces_have_zero_alignment(self):
        # Centered vectors with orthogonal column spaces yield zero linear CKA.
        x = torch.tensor([[1.0], [-1.0], [0.0], [0.0]])
        y = torch.tensor([[0.0], [0.0], [1.0], [-1.0]])
        geometry = Geometry()
        geometry.add(x, y)
        self.assertAlmostEqual(geometry.result()['window_linear_cka'], 0.0, places=6)

    def test_aggregation_excludes_undefined_windows_and_keeps_primary_counts(self):
        geometry = Geometry()
        reference = torch.randn(7, 5)
        geometry.add(reference*1e4, reference*1e4)
        x = torch.tensor([[1.0], [-1.0], [0.0], [0.0]])
        y = torch.tensor([[0.0], [0.0], [1.0], [-1.0]])
        geometry.add(x, y)
        geometry.add(torch.ones(3, 5), torch.ones(3, 5))
        result = geometry.result()
        self.assertEqual(result['tokens'], 14)
        self.assertEqual(result['valid_cka_tokens'], 11)
        self.assertAlmostEqual(result['window_linear_cka'], 7/11, places=6)

    def test_constant_and_single_token_covariance_is_undefined(self):
        for x, y in [(torch.ones(4, 3), torch.ones(4, 3)),
                     (torch.ones(4, 3), torch.randn(4, 3)),
                     (torch.randn(1, 3), torch.randn(1, 3))]:
            geometry = Geometry()
            geometry.add(x, y)
            result = geometry.result()
            self.assertIsNone(result['window_linear_cka'])
            self.assertEqual(result['valid_cka_tokens'], 0)
            self.assertEqual(result['tokens'], len(x))


if __name__ == '__main__':
    unittest.main()
