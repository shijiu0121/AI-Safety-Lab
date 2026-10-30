import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["AEGIS_DEVICE"] = "cpu"
import torch
from backend.training_config import TrainingConfig
from backend.training_data import prepare_data, DATASETS
from backend.train import distillation_loss, check_stop
from backend import engine
from fixture_support import fixture_resources


class ExtendedTrainingTests(unittest.TestCase):
    def test_distillation_loss_has_finite_student_gradient(self):
        student = torch.tensor([[1.0, -1.0]], requires_grad=True)
        teacher = torch.tensor([[2.0, -2.0]])
        loss = distillation_loss(student, teacher, torch.tensor([0]), 20, 0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(student.grad.abs().sum()), 0)
        equal = distillation_loss(teacher, teacher, torch.tensor([0]), 20, 0)
        self.assertAlmostEqual(float(equal), 0, places=4)

    def test_imagefolder_split_is_disjoint_and_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec, _ = fixture_resources(root)
            config = TrainingConfig(
                dataset="imagefolder",
                data_dir=str(root / "images"),
                mean=[0, 0, 0],
                std=[1, 1, 1],
                input_size=16,
                num_workers=0,
            ).model_dump()
            output = root / "training"
            output.mkdir()
            train, val, info = prepare_data(config, output)
            self.assertFalse(set(train.dataset.indices) & set(val.dataset.indices))
            self.assertEqual(info["classes"], ["A", "B"])
            self.assertFalse(
                json.loads((output / "split.json").read_text())["official_test_used"]
            )
            with self.assertRaisesRegex(ValueError, "变化"):
                prepare_data(dict(config, seed=7), output)

    def test_validation_directory_cannot_overlap_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_resources(root)
            config = TrainingConfig(
                dataset="imagefolder",
                data_dir=str(root / "images"),
                validation_dir=str(root / "images"),
                mean=[0, 0, 0],
                std=[1, 1, 1],
                input_size=16,
            ).model_dump()
            with self.assertRaisesRegex(ValueError, "相同文件"):
                prepare_data(config, root)

    def test_training_schema_rejects_invalid_distillation_and_cifar_split(self):
        with self.assertRaises(ValueError):
            TrainingConfig(method="distill")
        with self.assertRaises(ValueError):
            TrainingConfig(dataset="cifar10", validation_dir="/test")
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=float("nan"))
        self.assertTrue(callable(DATASETS["cifar10"]["class"]))

    def test_cooperative_stop_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "stop"
            before = os.environ.get("AEGIS_STOP_FILE")
            os.environ["AEGIS_STOP_FILE"] = str(marker)
            try:
                check_stop()
                marker.touch()
                with self.assertRaises(engine.Cancelled):
                    check_stop()
            finally:
                if before is None:
                    os.environ.pop("AEGIS_STOP_FILE", None)
                else:
                    os.environ["AEGIS_STOP_FILE"] = before
