import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["AEGIS_DEVICE"] = "cpu"
import torch
from PIL import Image
from backend import engine
from backend.batch_engine import BatchConfig, run_suite
from backend.data_sources import inspect_dataset


class BatchEngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="aegis-suite-test-")
        self.root = Path(self.temporary.name)
        image_root = self.root / "images"
        for class_name, value in [("A", 130), ("B", 120)]:
            directory = image_root / class_name
            directory.mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (16, 16), (value, value, value)).save(
                    directory / f"{index}.png"
                )
        loader = self.root / "model.py"
        loader.write_text(
            "from torch import nn\ndef build_model():\n    return nn.Sequential(nn.Flatten(), nn.Linear(768, 2))\n",
            encoding="utf-8",
        )
        torch.manual_seed(1)
        weights = self.root / "model.pt"
        torch.save(
            torch.nn.Sequential(
                torch.nn.Flatten(), torch.nn.Linear(768, 2)
            ).state_dict(),
            weights,
        )
        self.model = {
            "id": "fixture_model",
            "name": "测试网络",
            "architecture": "custom",
            "loader_path": str(loader),
            "factory": "build_model",
            "weights_path": str(weights),
            "num_classes": 2,
            "input_size": 16,
            "mean": [0, 0, 0],
            "std": [1, 1, 1],
            "categories": ["A", "B"],
            "status": "validated",
        }
        self.model["validation"] = engine.validate_model(self.model)
        self.dataset = {
            "id": "fixture_data",
            "name": "临时图片目录",
            "kind": "imagefolder",
            "root": str(image_root),
            "split": "test",
            "status": "validated",
        }
        self.dataset["inspection"] = inspect_dataset(self.dataset)

    def tearDown(self):
        self.temporary.cleanup()

    def configuration(self, **options):
        return BatchConfig(
            dataset_id="fixture_data",
            model_ids=["fixture_model"],
            attacks=["fgsm", "pgd20"],
            linf_budgets_255=[0, 8],
            batch_size=2,
            **options,
        )

    def test_batch_reports_and_union_are_per_sample(self):
        output = self.root / "results"
        report = run_suite(self.configuration(), self.dataset, [self.model], output)
        self.assertEqual(len(report["cells"]), 4)
        self.assertEqual(len(report["combined"]), 2)
        self.assertFalse(report["subset"])
        for combined in report["combined"]:
            individual = [
                cell["metrics"]["robust_accuracy"]
                for cell in report["cells"]
                if cell["protocol"]["budget"] == combined["budget"]
            ]
            self.assertLessEqual(
                combined["metrics"]["robust_accuracy"], min(individual)
            )
        for filename in (
            "summary.csv",
            "per_class.csv",
            "samples.csv",
            "manifest.json",
            "sample_indices.json",
        ):
            self.assertTrue((output / filename).is_file())
        repeated = run_suite(self.configuration(), self.dataset, [self.model], output)
        self.assertTrue(all(cell["reused_batches"] == 2 for cell in repeated["cells"]))

    def test_cancel_then_resume_reuses_only_completed_batches(self):
        output = self.root / "partial"
        stopped = [False]

        def progress(value, message, report):
            stopped[0] = True

        with self.assertRaises(engine.Cancelled):
            run_suite(
                self.configuration(),
                self.dataset,
                [self.model],
                output,
                progress,
                lambda: stopped[0],
            )
        report = run_suite(self.configuration(), self.dataset, [self.model], output)
        self.assertEqual(report["cells"][0]["reused_batches"], 1)

    def test_changed_protocol_refuses_checkpoint(self):
        output = self.root / "protocol"
        run_suite(self.configuration(), self.dataset, [self.model], output)
        with self.assertRaisesRegex(ValueError, "恢复条件"):
            run_suite(self.configuration(seed=10), self.dataset, [self.model], output)

    def test_subsets_are_fixed_and_explicit(self):
        report = run_suite(
            self.configuration(sample_limit=3),
            self.dataset,
            [self.model],
            self.root / "subset",
        )
        self.assertEqual(report["sample_count"], 3)
        self.assertTrue(report["subset"])
        self.assertEqual(
            len(json.loads((self.root / "subset" / "sample_indices.json").read_text())),
            3,
        )


if __name__ == "__main__":
    unittest.main()
