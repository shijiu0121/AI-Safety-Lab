"""新工作流的实际接口检查，使用隔离数据库和手工网络。"""

import io
import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ["AEGIS_DEVICE"] = "cpu"
from PIL import Image
from fastapi.testclient import TestClient
from backend import store
from backend.app import app
from fixture_support import fixture_resources


class WorkflowApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="aegis-workflow-api-")
        cls.root = Path(cls.temporary.name)
        cls.original_data, cls.original_db = store.DATA, store.DB
        store.DATA = cls.root / "records"
        store.DATA.mkdir()
        store.DB = store.DATA / "test.sqlite3"
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()
        cls.spec, cls.dataset = fixture_resources(cls.root / "fixtures")
        store.save("models", cls.spec)
        store.save("datasets", cls.dataset)
        cls.profile = cls.client.post(
            "/api/defenses",
            json=dict(
                name="接口功能防御",
                model_ids=[cls.spec["id"]],
                transforms=[dict(kind="median")],
                detector=dict(method="confidence", threshold=0.9),
            ),
        ).json()
        image = Image.new("RGB", (16, 16), (130, 130, 130))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        cls.image = buffer.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)
        store.DATA, store.DB = cls.original_data, cls.original_db
        cls.temporary.cleanup()

    def wait(self, kind, key):
        for _ in range(400):
            response = self.client.get(f"/api/workflows/{kind}/{key}")
            self.assertEqual(response.status_code, 200, response.text)
            job = response.json()
            if job["status"] not in ("queued", "running", "cancelling"):
                return job
            time.sleep(0.02)
        self.fail("功能测试任务超时。")

    def test_calibration_rejects_test_split_and_apply_is_explicit(self):
        response = self.client.post(
            "/api/calibrations",
            json=dict(defense_id=self.profile["id"], dataset_id=self.dataset["id"]),
        )
        self.assertEqual(response.status_code, 422)
        validation = dict(self.dataset, id="validation_fixture", split="validation")
        store.save("datasets", validation)
        response = self.client.post(
            "/api/calibrations",
            json=dict(
                defense_id=self.profile["id"], dataset_id=validation["id"], batch_size=2
            ),
        )
        self.assertEqual(response.status_code, 202, response.text)
        job = self.wait("calibrations", response.json()["id"])
        self.assertEqual(job["status"], "succeeded", job)
        before = store.get("defenses", self.profile["id"])
        self.assertIsNone(before["calibration"])
        response = self.client.post(f'/api/calibrations/{job["id"]}/apply')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["config"]["detector"]["threshold"],
            job["report"]["threshold"],
        )
        self.assertIsNotNone(response.json()["calibration"])

    def test_defense_job_report_files_and_completed_resume_rejection(self):
        response = self.client.post(
            "/api/defense-runs",
            json=dict(
                dataset_id=self.dataset["id"],
                defense_ids=[self.profile["id"]],
                attacks=["pgd20"],
                modes=["transfer"],
                batch_size=2,
            ),
        )
        self.assertEqual(response.status_code, 202, response.text)
        job = self.wait("defense_runs", response.json()["id"])
        self.assertEqual(job["status"], "succeeded", job)
        for name in (
            "report.json",
            "report.html",
            "summary.csv",
            "per_class.csv",
            "samples.csv",
            "manifest.json",
        ):
            download = self.client.get(
                f'/api/workflows/defense_runs/{job["id"]}/files/{name}'
            )
            self.assertEqual(download.status_code, 200, name)
        response = self.client.post(f'/api/workflows/defense_runs/{job["id"]}/resume')
        self.assertEqual(response.status_code, 409)
        response = self.client.get(
            f'/api/workflows/defense_runs/{job["id"]}/files/%2E%2E%2Faegis.sqlite3'
        )
        self.assertEqual(response.status_code, 404)

    def test_realtime_monitor_reject_and_invalid_upload_logging(self):
        profile = self.client.post(
            "/api/defenses",
            json=dict(
                name="输入异常检查",
                model_ids=[self.spec["id"]],
                detector=dict(min_input_std=0.01),
            ),
        ).json()
        for policy, decision in [("monitor", "allow"), ("reject", "reject")]:
            session = self.client.post(
                "/api/guard/sessions",
                json=dict(
                    name="接口测试",
                    defense_id=profile["id"],
                    policy=policy,
                    save_images=True,
                ),
            ).json()
            response = self.client.post(
                f'/api/guard/sessions/{session["id"]}/predict',
                files={"image": ("test.png", self.image, "image/png")},
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["decision"], decision)
            self.assertTrue(result["flagged"])
            self.assertEqual(result["prediction"] is None, policy == "reject")
            self.assertEqual(
                self.client.get(
                    f'/api/guard/events/{result["id"]}/images/processed'
                ).status_code,
                200,
            )
            self.client.post(f'/api/guard/sessions/{session["id"]}/stop')
            self.assertEqual(
                self.client.post(
                    f'/api/guard/sessions/{session["id"]}/predict',
                    files={"image": ("test.png", self.image)},
                ).status_code,
                409,
            )
        session = self.client.post(
            "/api/guard/sessions",
            json=dict(name="非法图片测试", defense_id=profile["id"]),
        ).json()
        response = self.client.post(
            f'/api/guard/sessions/{session["id"]}/predict',
            files={"image": ("bad.png", b"invalid")},
        )
        self.assertEqual(response.status_code, 422)
        stats = self.client.get(
            "/api/guard/events", params={"session_id": session["id"]}
        ).json()
        self.assertEqual(stats["errors"], 1)
        self.client.post(f'/api/guard/sessions/{session["id"]}/stop')

    def test_resource_import_edit_archive_restore_and_metadata_reset(self):
        response = self.client.post(
            "/api/resources/import", json=dict(kind="models", resource=self.spec)
        )
        self.assertEqual(response.status_code, 201, response.text)
        model = response.json()
        self.assertEqual(model["status"], "registered")
        self.assertIsNone(model["validation"])
        exported = self.client.get(f'/api/resources/models/{model["id"]}/export').json()
        exported["name"] = "修改后的测试模型"
        response = self.client.put(f'/api/models/{model["id"]}', json=exported)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.client.post(
                f'/api/resources/models/{model["id"]}/archive'
            ).status_code,
            200,
        )
        self.assertNotIn(
            model["id"], [item["id"] for item in self.client.get("/api/models").json()]
        )
        self.assertEqual(
            self.client.post(
                f'/api/resources/models/{model["id"]}/restore'
            ).status_code,
            200,
        )

    def test_local_transform_preview_is_real_and_bounded(self):
        response = self.client.post(
            "/api/transform-preview",
            data={"config": '[{"kind":"bit_depth","bits":2}]'},
            files={"image": ("image.png", self.image, "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        with Image.open(io.BytesIO(response.content)) as result:
            self.assertEqual(result.getpixel((0, 0)), (170, 170, 170))

    def test_training_can_be_configured_but_never_runs_on_cpu(self):
        response = self.client.post(
            "/api/training-runs",
            json=dict(
                name="只验证配置", config=dict(method="erm", epochs=1, num_workers=0)
            ),
        )
        self.assertEqual(response.status_code, 201, response.text)
        job = response.json()
        self.assertEqual(job["status"], "configured")
        response = self.client.post(f'/api/training-runs/{job["id"]}/start')
        self.assertEqual(response.status_code, 422)
        self.assertEqual(store.get("training_runs", job["id"])["status"], "configured")

    def test_workspace_summary_counts_all_completed_workflows(self):
        response = self.client.get("/api/workspace-summary")
        self.assertEqual(response.status_code, 200, response.text)
        summary = response.json()
        self.assertIn("defense_runs", summary["kinds"])
        self.assertIn("training_runs", summary["kinds"])
        self.assertEqual(
            summary["completed"],
            sum(item["completed"] for item in summary["kinds"].values()),
        )
        self.assertEqual(
            summary["kinds"]["training_runs"]["total"],
            len(store.all_items("training_runs")),
        )

    def test_cli_report_registration_is_visible_and_downloadable(self):
        import shutil
        from backend.plan_results import publish_evaluation
        from backend.defense_evaluation import DefenseEvaluationConfig

        response = self.client.post(
            "/api/defense-runs",
            json=dict(
                dataset_id=self.dataset["id"],
                defense_ids=[self.profile["id"]],
                attacks=["fgsm"],
                modes=["transfer"],
                batch_size=2,
            ),
        )
        self.assertEqual(response.status_code, 202, response.text)
        original = self.wait("defense_runs", response.json()["id"])
        self.assertEqual(original["status"], "succeeded", original)
        key = "cli-published-functional-check"
        shutil.copytree(store.DATA / original["id"], store.DATA / key)
        config = DefenseEvaluationConfig.model_validate(original["config"])
        publish_evaluation(
            "defense_runs",
            key,
            config,
            self.dataset,
            original["report"],
            [self.spec],
            "2026-10-29T00:00:00+00:00",
        )
        detail = self.client.get("/api/workflows/defense_runs/" + key).json()
        self.assertEqual(detail["report"]["sample_count"], 6)
        self.assertEqual(
            self.client.get(
                "/api/workflows/defense_runs/" + key + "/files/report.html"
            ).status_code,
            200,
        )

    def test_cli_log_capture_runs_only_print_fixture(self):
        import sys
        from backend.plan_results import run_logged_training

        key = "cli-log-functional-check"
        run_logged_training(
            [sys.executable, "-c", 'print("CLI functional log fixture; no training")'],
            key,
        )
        self.assertIn(
            "no training",
            (store.DATA / key / "training.log").read_text(encoding="utf-8"),
        )
