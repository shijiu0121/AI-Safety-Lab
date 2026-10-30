"""小型网络功能检查，不代表正式数据集或 GPU 鲁棒性结果。"""

import os
import unittest
from unittest.mock import patch

os.environ["AEGIS_DEVICE"] = "cpu"
import torch
from backend.attack_protocol import AttackConfig, check_attack_compatibility
from backend.attacks import run_attack, fixed_random_state


class BoundaryClassifier(torch.nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.classes = classes

    def forward(self, images):
        center = images.mean((1, 2, 3)) - 0.5
        scores = [center, -center] + [
            torch.full_like(center, -2) for _ in range(self.classes - 2)
        ]
        return torch.stack(scores, 1)


class NewAttackTests(unittest.TestCase):
    def setUp(self):
        self.images = torch.full((2, 3, 8, 8), 0.51)
        self.labels = torch.zeros(2, dtype=torch.long)
        self.spec = {"num_classes": 2, "mean": [0, 0, 0], "std": [1, 1, 1]}
        self.model = BoundaryClassifier().eval()

    def test_pgd_preset_has_exact_iteration_count(self):
        for name, count in [("pgd20", 20), ("pgd100", 100)]:
            config = AttackConfig(algorithm=name, steps=3, epsilon_255=8)
            _, details = run_attack(
                self.model, self.images, self.labels, self.spec, config
            )
            self.assertEqual(details["protocol"]["iterations"], count)
            self.assertEqual(len(details["trace"]), count)

    def test_cw_linf_uses_margin_and_preserves_success(self):
        config = AttackConfig(algorithm="cw_linf", cw_steps=30, epsilon_255=8)
        adversarial, details = run_attack(
            self.model, self.images, self.labels, self.spec, config
        )
        self.assertTrue(self.model(adversarial).argmax(1).ne(self.labels).all())
        self.assertLessEqual(
            float((adversarial - self.images).abs().max()), 8 / 255 + 1e-6
        )
        self.assertEqual(details["protocol"]["norm"], "Linf")
        self.assertIn("待原实验代码", details["protocol"]["paper_alignment"])

    def test_real_foolbox_deepfool_both_norms(self):
        for name in ("deepfool_linf", "deepfool_l2"):
            config = AttackConfig(
                algorithm=name,
                epsilon_255=8,
                epsilon_l2=0.5,
                deepfool_steps=5,
                deepfool_candidates=2,
            )
            adversarial, details = run_attack(
                self.model, self.images, self.labels, self.spec, config
            )
            self.assertEqual(adversarial.shape, self.images.shape)
            self.assertEqual(details["protocol"]["backend"], "Foolbox")
            self.assertEqual(details["trace"], [])

    def test_official_autoattack_standard_small_network_smoke(self):
        # 使用官方未删减 standard；套件可在样本被早期攻击成功后自动跳过剩余计算。
        model = BoundaryClassifier(classes=10).eval()
        spec = dict(self.spec, num_classes=10)
        config = AttackConfig(algorithm="autoattack", epsilon_255=8, seed=0)
        adversarial, details = run_attack(
            model, self.images[:1], self.labels[:1], spec, config
        )
        self.assertEqual(
            details["protocol"]["components"], ["apgd-ce", "apgd-t", "fab-t", "square"]
        )
        self.assertEqual(details["protocol"]["autoattack_version"], "standard")
        self.assertLessEqual(
            float((adversarial - self.images[:1]).abs().max()), 8 / 255 + 1e-5
        )

    def test_small_class_count_does_not_silently_reduce_aa(self):
        with self.assertRaisesRegex(ValueError, "至少 10"):
            check_attack_compatibility(AttackConfig(algorithm="autoattack"), 2)

    def test_external_attack_cancellation(self):
        from backend.engine import Cancelled

        with self.assertRaises(Cancelled):
            run_attack(
                self.model,
                self.images,
                self.labels,
                self.spec,
                AttackConfig(algorithm="deepfool_linf"),
                cancelled=lambda: True,
            )

    def test_external_rng_state_is_restored(self):
        state = torch.get_rng_state().clone()
        with fixed_random_state(123):
            torch.rand(4)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
