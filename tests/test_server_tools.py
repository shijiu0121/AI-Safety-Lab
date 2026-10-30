import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["AEGIS_DEVICE"] = "cpu"
import torch
from backend import engine
from backend.training_config import TrainingConfig
from backend.training_data import prepare_data
from backend.training_state import atomic_tensor_save, publish_outputs
from backend.export_validation import export_validation
from backend.server_check import check_environment
from backend.reporting import write_html_report
from fixture_support import fixture_resources


class ServerToolTests(unittest.TestCase):
    def test_checkpoint_can_rebuild_outputs_without_changing_weights_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec, _ = fixture_resources(root)
            state = torch.load(spec["weights_path"], weights_only=True)
            checkpoint = dict(
                config=dict(dataset="imagefolder", method="erm"),
                best_state_dict=state,
                model_config={
                    key: spec[key]
                    for key in (
                        "architecture",
                        "num_classes",
                        "input_size",
                        "mean",
                        "std",
                        "categories",
                        "loader_path",
                        "factory",
                    )
                },
                history=[dict(epoch=1, training_loss=1.0)],
            )
            atomic_tensor_save(checkpoint, root / "last.pt")
            publish_outputs(checkpoint, root)
            before = engine.sha256(root / "best_weights.pt")
            (root / "metrics.jsonl").write_text("incomplete metrics", encoding="utf-8")
            publish_outputs(torch.load(root / "last.pt", weights_only=True), root)
            self.assertEqual(before, engine.sha256(root / "best_weights.pt"))
            self.assertEqual(
                json.loads((root / "metrics.jsonl").read_text())["epoch"], 1
            )
            self.assertTrue((root / "model_registration.json").is_file())

    def test_validation_export_uses_only_reserved_indices_and_preserves_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_resources(root)
            config = TrainingConfig(
                dataset="imagefolder",
                data_dir=str(root / "images"),
                input_size=16,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                num_workers=0,
            ).model_dump()
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            training = root / "training"
            training.mkdir()
            _, val, _ = prepare_data(config, training)
            exported = export_validation(
                config_path, training / "split.json", root / "validation-export"
            )
            pngs = list(Path(exported["root"]).rglob("*.png"))
            self.assertEqual(len(pngs), len(val.dataset.indices))
            before = {str(path): path.stat().st_mtime_ns for path in pngs}
            export_validation(
                config_path, training / "split.json", root / "validation-export"
            )
            self.assertEqual(
                before, {str(path): path.stat().st_mtime_ns for path in pngs}
            )
            with self.assertRaisesRegex(ValueError, "源数据"):
                export_validation(
                    config_path, training / "split.json", root / "images" / "bad"
                )

    def test_sample_caps_keep_classes_and_are_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_resources(root)
            config = TrainingConfig(
                dataset="imagefolder",
                data_dir=str(root / "images"),
                input_size=16,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                num_workers=0,
                max_train_samples=2,
                max_validation_samples=2,
            ).model_dump()
            output = root / "training"
            output.mkdir()
            train, val, _ = prepare_data(config, output)
            self.assertEqual(len(train.dataset), 2)
            self.assertEqual(len(val.dataset), 2)
            self.assertEqual(
                json.loads((output / "split.json").read_text())["class_counts"], [1, 1]
            )

    def test_cpu_diagnostic_never_claims_gpu_validation(self):
        result = check_environment(cpu_only=True)
        self.assertTrue(result["ok"])
        self.assertFalse(result["gpu_test_executed"])
        self.assertIn("CPU", result["scope"])

    def test_native_architectures_output_contract_and_input_gradient(self):
        for name in engine.ARCHITECTURES:
            with self.subTest(architecture=name):
                model = engine.ARCHITECTURES[name](weights=None, num_classes=3).eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                images = torch.rand(1, 3, 32, 32, requires_grad=True)
                outputs = model(images)
                self.assertEqual(tuple(outputs.shape), (1, 3))
                gradient = torch.autograd.grad(outputs[:, 0].sum(), images)[0]
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                del model, images, outputs, gradient

    def test_html_escapes_user_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.html"
            write_html_report(
                dict(
                    dataset_name="<script>bad()</script>", cells=[], note="<b>text</b>"
                ),
                path,
            )
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("<script>", source)
            self.assertIn("&lt;script&gt;", source)
