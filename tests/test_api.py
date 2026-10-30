"""使用临时小网络验证端到端接口；不使用正式权重，也不启动训练。"""

import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

TEMPORARY = tempfile.TemporaryDirectory(prefix="aegis-test-")
os.environ["AEGIS_DATA_DIR"] = str(Path(TEMPORARY.name) / "data")
os.environ["AEGIS_DEVICE"] = "cpu"

import torch
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()
        root = Path(TEMPORARY.name)
        cls.loader = root / "fixture_model.py"
        cls.loader.write_text(
            "from torch import nn\ndef build_model():\n    return nn.Sequential(nn.Flatten(), nn.Linear(3*16*16, 2))\n",
            encoding="utf-8",
        )
        cls.weights = root / "fixture_weights.pt"
        torch.manual_seed(0)
        torch.save(
            torch.nn.Sequential(
                torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 2)
            ).state_dict(),
            cls.weights,
        )
        payload = {
            "name": "仅用于接口测试",
            "architecture": "custom",
            "loader_path": str(cls.loader),
            "weights_path": str(cls.weights),
            "num_classes": 2,
            "input_size": 16,
            "mean": [0, 0, 0],
            "std": [1, 1, 1],
            "categories": ["A", "B"],
        }
        response = cls.client.post("/api/models", json=payload)
        assert response.status_code == 201, response.text
        cls.model_id = response.json()["id"]
        response = cls.client.post(f"/api/models/{cls.model_id}/validate")
        assert response.status_code == 200, response.text
        image = Image.new("RGB", (16, 16), color=(120, 120, 120))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        cls.image = buffer.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        TEMPORARY.cleanup()

    def wait_for_task(self, task_id):
        for _ in range(200):
            task = self.client.get(f"/api/experiments/{task_id}").json()
            if task["status"] not in ("queued", "running", "cancelling"):
                return task
            time.sleep(0.02)
        self.fail("任务未在测试时限内完成。")

    def create_task(self, **extra):
        config = {
            "model_id": self.model_id,
            "algorithm": "pgd",
            "steps": 3,
            "epsilon_255": 8,
            "step_size_255": 2,
            **extra,
        }
        return self.client.post(
            "/api/experiments",
            data={"config": json.dumps(config)},
            files={"image": ("fixture.png", self.image, "image/png")},
        )

    def test_real_attack_and_exports_without_ground_truth(self):
        response = self.create_task()
        self.assertEqual(response.status_code, 202, response.text)
        task = self.wait_for_task(response.json()["id"])
        self.assertEqual(task["status"], "succeeded", task)
        self.assertIsNone(task["result"]["attack_success"])
        self.assertEqual(task["result"]["label_source"], "original_prediction")
        self.assertLessEqual(task["result"]["linf"], 8 / 255 + 1e-6)
        self.assertEqual(len(task["result"]["trace"]), 3)
        self.assertIn("weights_sha256", task["model_snapshot"])
        for suffix in (
            "images/adversarial",
            "images/perturbation",
            "report",
            "tensors/adversarial",
        ):
            exported = self.client.get(f"/api/experiments/{task['id']}/{suffix}")
            self.assertEqual(exported.status_code, 200)
            self.assertGreater(len(exported.content), 0)

    def test_initially_wrong_label_is_not_counted_as_attack_success(self):
        first = self.create_task(algorithm="fgsm")
        first_task = self.wait_for_task(first.json()["id"])
        original_class = first_task["result"]["clean_top5"][0]["index"]
        second = self.create_task(true_label=1 - original_class)
        task = self.wait_for_task(second.json()["id"])
        self.assertEqual(task["status"], "succeeded", task)
        self.assertFalse(task["result"]["originally_correct"])
        self.assertIsNone(task["result"]["attack_success"])

    def test_invalid_label_is_rejected_before_queue(self):
        self.assertEqual(self.create_task(true_label=10).status_code, 422)

    def test_invalid_image_is_rejected(self):
        config = {"model_id": self.model_id, "algorithm": "fgsm"}
        response = self.client.post(
            "/api/experiments",
            data={"config": json.dumps(config)},
            files={"image": ("bad.png", b"not an image", "image/png")},
        )
        self.assertEqual(response.status_code, 422)

    def test_invalid_normalization_is_rejected(self):
        response = self.client.post(
            "/api/models",
            json={
                "name": "bad",
                "architecture": "resnet18",
                "weights_path": "none",
                "std": [0, 1, 1],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_dataset_and_batch_routes(self):
        root = Path(TEMPORARY.name) / "api-dataset"
        for name in ("A", "B"):
            folder = root / name
            folder.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (120, 120, 120)).save(folder / "sample.png")
        response = self.client.post(
            "/api/datasets",
            json={
                "name": "接口测试目录",
                "kind": "imagefolder",
                "root": str(root),
                "split": "test",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        dataset_id = response.json()["id"]
        response = self.client.post(f"/api/datasets/{dataset_id}/inspect")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["inspection"]["sample_count"], 2)
        response = self.client.post(
            "/api/batch-runs",
            json={
                "dataset_id": dataset_id,
                "model_ids": [self.model_id],
                "attacks": ["pgd20", "cw_linf", "deepfool_linf"],
                "linf_budgets_255": [8],
                "batch_size": 2,
                "cw_steps": 3,
                "deepfool_steps": 3,
            },
        )
        self.assertEqual(response.status_code, 202, response.text)
        key = response.json()["id"]
        for _ in range(500):
            job = self.client.get(f"/api/batch-runs/{key}").json()
            if job["status"] not in ("queued", "running", "cancelling"):
                break
            time.sleep(0.02)
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual(len(job["report"]["cells"]), 3)
        self.assertEqual(
            self.client.get(f"/api/batch-runs/{key}/files/summary.csv").status_code, 200
        )
        self.assertEqual(
            self.client.post(f"/api/batch-runs/{key}/resume").status_code, 409
        )
        self.assertEqual(
            self.client.get(
                f"/api/batch-runs/{key}/files/%2e%2e%2f..%2fREADME.md"
            ).status_code,
            404,
        )
        protocols = self.client.get("/api/attacks").json()
        self.assertTrue(
            all(
                name in [entry["id"] for entry in protocols]
                for name in (
                    "pgd20",
                    "pgd100",
                    "cw_linf",
                    "autoattack",
                    "deepfool_linf",
                )
            )
        )

    def test_missing_experiment(self):
        self.assertEqual(
            self.client.get("/api/experiments/not-a-task").status_code, 404
        )


if __name__ == "__main__":
    unittest.main()
