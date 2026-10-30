"""CPU 数学检查，不代表 GPU 性能或正式数据集实验。"""

import os
import unittest
from unittest.mock import patch

os.environ["AEGIS_DEVICE"] = "cpu"
import torch
from backend.engine import attack_tensor, forward, select_device


class MeanClassifier(torch.nn.Module):
    def forward(self, images):
        average = images.mean(dim=(1, 2, 3))
        return torch.stack((average, -average), dim=1)


class AttackTests(unittest.TestCase):
    def setUp(self):
        self.model = MeanClassifier().eval()
        self.spec = {"num_classes": 2, "mean": [0, 0, 0], "std": [1, 1, 1]}
        self.images = torch.full((2, 3, 16, 16), 0.5)
        self.labels = torch.zeros(2, dtype=torch.long)

    def test_fgsm_matches_analytic_gradient(self):
        attacked, trace = attack_tensor(
            self.model, self.images, self.labels, self.spec, "fgsm", 0.1, 1, 0.1, 0
        )
        self.assertTrue(torch.allclose(attacked, torch.full_like(attacked, 0.4)))
        self.assertEqual(len(trace), 1)

    def test_zero_budget_does_not_change_input(self):
        attacked, _ = attack_tensor(
            self.model, self.images, self.labels, self.spec, "pgd", 0, 3, 0.1, 0
        )
        self.assertTrue(torch.equal(attacked, self.images))

    def test_pgd_projection_and_seed(self):
        attacked, _ = attack_tensor(
            self.model, self.images, self.labels, self.spec, "pgd", 0.08, 4, 0.02, 7
        )
        repeated, _ = attack_tensor(
            self.model, self.images, self.labels, self.spec, "pgd", 0.08, 4, 0.02, 7
        )
        self.assertLessEqual(float((attacked - self.images).abs().max()), 0.080001)
        self.assertTrue(torch.equal(attacked, repeated))
        self.assertGreaterEqual(float(attacked.min()), 0)
        self.assertLessEqual(float(attacked.max()), 1)

    def test_pixel_bounds_at_boundary(self):
        images = torch.zeros_like(self.images)
        attacked, _ = attack_tensor(
            self.model, images, self.labels, self.spec, "fgsm", 0.1, 1, 0.1, 0
        )
        self.assertTrue(torch.equal(attacked, images))

    def test_batch_attack_without_trace(self):
        attacked, trace = attack_tensor(
            self.model,
            self.images,
            self.labels,
            self.spec,
            "pgd",
            0.1,
            2,
            0.02,
            7,
            record_trace=False,
        )
        self.assertEqual(attacked.shape, self.images.shape)
        self.assertEqual(trace, [])

    def test_requested_cuda_never_silently_falls_back(self):
        with patch.dict(os.environ, {"AEGIS_DEVICE": "cuda"}), patch(
            "torch.cuda.is_available", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                select_device()

    def test_bad_model_output_is_rejected(self):
        wrong_spec = dict(self.spec, num_classes=3)
        with self.assertRaisesRegex(ValueError, "logits"):
            forward(self.model, self.images, wrong_spec)


if __name__ == "__main__":
    unittest.main()
