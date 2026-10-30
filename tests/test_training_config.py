"""只检查训练配置与损失构造，不运行数据集训练。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from backend.train import load_configuration, generate_training_attack, balanced_softmax_loss, hierarchical_equalization_loss, train_one_epoch
from backend.training_config import TrainingConfig


class TrainingConfigurationTests(unittest.TestCase):
    def test_at_bsl_loss_uses_class_frequencies(self):
        logits = torch.tensor([[0.3, -0.5], [1.2, -0.1]], requires_grad=True)
        labels = torch.tensor([0, 1])
        actual = balanced_softmax_loss(logits, labels, [10, 2])
        expected = torch.nn.functional.cross_entropy(logits + torch.tensor([10., 2.]).log(), labels)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_taet_loss_matches_classwise_notebook_formula(self):
        logits = torch.tensor([[1., 0., -1.], [0.2, 1., 0.]], requires_grad=True)
        labels = torch.tensor([0, 1])
        losses = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        class_losses = torch.stack([losses[0], losses[1], losses[0] * 0])
        avg = class_losses.mean()
        expected = .1 * avg + .1 * (class_losses - avg).square().mean() + .1 * (class_losses / class_losses.sum()).square().sum()
        actual = hierarchical_equalization_loss(logits, labels)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_new_methods_run_one_toy_batch(self):
        images = torch.rand(4, 3, 16, 16)
        labels = torch.tensor([0, 0, 0, 1])
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=4)
        spec = {"num_classes": 2, "mean": [0, 0, 0], "std": [1, 1, 1]}
        for method, epoch in [("at_bsl", 0), ("taet", 0), ("taet", 1)]:
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 2))
            config = TrainingConfig(method=method, taet_warmup_epochs=1, attack_steps=2).model_dump()
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            loss = train_one_epoch(model, loader, optimizer, spec, config, torch.device('cpu'), epoch, class_counts=[3, 1])
            self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_unknown_parameter_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"epoch_typo": 20}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "未知配置"):
                load_configuration(path)

    def test_pgd_and_trades_attack_preserve_training_mode_and_bounds(self):
        model = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 2)
        ).train()
        spec = {"num_classes": 2, "mean": [0, 0, 0], "std": [1, 1, 1]}
        images = torch.full((3, 3, 16, 16), 0.5)
        labels = torch.tensor([0, 1, 0])
        for method in ("pgd_at", "trades"):
            config = {
                "method": method,
                "epsilon_255": 8,
                "attack_steps": 2,
                "step_size_255": 2,
            }
            attacked = generate_training_attack(model, images, labels, spec, config, 0)
            self.assertTrue(model.training)
            self.assertFalse(attacked.requires_grad)
            self.assertLessEqual(float((attacked - images).abs().max()), 8 / 255 + 1e-6)


if __name__ == "__main__":
    unittest.main()
