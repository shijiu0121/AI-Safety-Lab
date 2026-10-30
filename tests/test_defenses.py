import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["AEGIS_DEVICE"] = "cpu"
import torch
from backend import engine
from backend.defenses import (
    DefenseConfig,
    DefenseRuntime,
    TransformConfig,
    transform_images,
)
from backend.defense_evaluation import (
    DefenseEvaluationConfig,
    run_defense_suite,
    adaptive_projected_attack,
    calibrate,
    score_roc,
)
from backend.attack_protocol import AttackConfig
from fixture_support import fixture_resources


class DefenseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="aegis-defense-test-")
        self.root = Path(self.temporary.name)
        self.spec, self.dataset = fixture_resources(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, **options):
        return DefenseConfig(name="测试防御", model_ids=[self.spec["id"]], **options)

    def test_transform_shape_bounds_and_explicit_bpda_gradient(self):
        x = torch.rand(2, 3, 16, 16, requires_grad=True)
        for kind in ("bit_depth", "median", "gaussian", "resize_pad", "noise"):
            transformed = transform_images(
                x,
                [TransformConfig(kind=kind)],
                torch.Generator().manual_seed(0),
                bpda=True,
            )
            self.assertEqual(transformed.shape, x.shape)
            self.assertTrue(bool(((transformed >= 0) & (transformed <= 1)).all()))
        quantized = transform_images(
            x, [TransformConfig(kind="bit_depth", bits=2)], bpda=True
        )
        self.assertTrue(torch.allclose(quantized * 3, (quantized * 3).round()))
        self.assertTrue(
            torch.allclose(
                torch.autograd.grad(quantized.sum(), x)[0], torch.ones_like(x)
            )
        )

    def test_stochastic_forward_and_eot_gradients_are_reproducible(self):
        runtime = DefenseRuntime(
            self.config(transforms=[dict(kind="noise")]), [self.spec]
        )
        x = torch.full((2, 3, 16, 16), 0.5, requires_grad=True)
        first = runtime.logits(x, 3)
        self.assertTrue(torch.equal(first, runtime.logits(x, 3)))
        self.assertFalse(torch.equal(first, runtime.logits(x, 4)))
        gradient = torch.autograd.grad(runtime.attack_logits(x)[:, 0].sum(), x)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_vote_strategies_and_detector_scores(self):
        x = torch.full((2, 3, 16, 16), 130 / 255)
        for aggregation in ("soft_vote", "logit_mean", "hard_vote", "random_member"):
            runtime = DefenseRuntime(self.config(aggregation=aggregation), [self.spec])
            self.assertEqual(runtime.measure(x)["prediction"].tolist(), [0, 0])
        for method in ("confidence", "entropy", "consistency"):
            runtime = DefenseRuntime(
                self.config(detector=dict(method=method, threshold=0)), [self.spec]
            )
            measured = runtime.measure(x)
            self.assertTrue(bool(torch.isfinite(measured["score"]).all()))
        runtime = DefenseRuntime(
            self.config(detector=dict(min_input_std=0.01)), [self.spec]
        )
        self.assertTrue(bool(runtime.measure(x)["anomaly"].all()))

    def test_adaptive_attack_respects_budget_and_cancellation(self):
        runtime = DefenseRuntime(
            self.config(transforms=[dict(kind="bit_depth", bits=4)]), [self.spec]
        )
        x = torch.full((2, 3, 16, 16), 130 / 255)
        labels = torch.zeros(2, dtype=torch.long)
        adversarial, details = adaptive_projected_attack(
            runtime, x, labels, AttackConfig(algorithm="pgd20"), False, 1
        )
        self.assertLessEqual(float((adversarial - x).abs().max()), 8 / 255 + 1e-6)
        self.assertTrue(details["protocol"]["bpda_identity"])
        self.assertTrue(
            bool(runtime.measure(adversarial)["prediction"].ne(labels).any())
        )
        with self.assertRaises(engine.Cancelled):
            adaptive_projected_attack(
                runtime,
                x,
                labels,
                AttackConfig(algorithm="cw_linf"),
                True,
                1,
                lambda: True,
            )

    def test_calibration_rejects_test_and_records_validation_provenance(self):
        profile = dict(
            id="profile",
            config=self.config(detector=dict(method="confidence")).model_dump(),
        )
        with self.assertRaisesRegex(ValueError, "validation"):
            calibrate(profile, [self.spec], self.dataset, self.root / "rejected")
        result = calibrate(
            profile,
            [self.spec],
            dict(self.dataset, split="validation"),
            self.root / "calibrated",
            target_fpr=0.1,
            batch_size=2,
        )
        self.assertLessEqual(result["empirical_fpr"], 0.1)
        self.assertEqual(len(result["input_sha256"]), 3)
        self.assertEqual(result["dataset_split"], "validation")

    def test_defense_report_resume_and_rejection_not_counted_correct(self):
        config = self.config(
            detector=dict(method="confidence", threshold=0),
            transforms=[dict(kind="median")],
        )
        profile = dict(
            id="profile",
            name="测试防御",
            model_ids=[self.spec["id"]],
            config=config.model_dump(),
        )
        options = DefenseEvaluationConfig(
            dataset_id=self.dataset["id"],
            defense_ids=["profile"],
            attacks=["pgd20"],
            modes=["transfer", "adaptive"],
            batch_size=2,
        )
        output = self.root / "evaluation"
        report = run_defense_suite(
            options, self.dataset, [profile], {"profile": [self.spec]}, output
        )
        self.assertEqual(len(report["cells"]), 2)
        for cell in report["cells"]:
            self.assertEqual(cell["detection"]["adversarial_coverage"], 0)
            self.assertEqual(cell["detection"]["accepted_correct_rate"], 0)
            self.assertIsNone(cell["detection"]["accepted_adversarial_accuracy"])
        repeated = run_defense_suite(
            options, self.dataset, [profile], {"profile": [self.spec]}, output
        )
        self.assertTrue(all(cell["reused_batches"] == 3 for cell in repeated["cells"]))
        self.assertTrue((output / "report.html").is_file())
        self.assertTrue((output / "per_class.csv").is_file())
        with self.assertRaisesRegex(ValueError, "恢复"):
            run_defense_suite(
                options.model_copy(update={"seed": 2}),
                self.dataset,
                [profile],
                {"profile": [self.spec]},
                output,
            )

    def test_aa_not_mislabeled_for_bpda_defense(self):
        profile = dict(
            id="profile",
            name="测试",
            config=self.config(transforms=[dict(kind="bit_depth")]).model_dump(),
        )
        options = DefenseEvaluationConfig(
            dataset_id=self.dataset["id"],
            defense_ids=["profile"],
            attacks=["autoattack"],
        )
        with self.assertRaisesRegex(ValueError, "自适应"):
            run_defense_suite(
                options,
                self.dataset,
                [profile],
                {"profile": [self.spec]},
                self.root / "invalid",
            )

    def test_roc_handles_ties_and_zero_perturbations(self):
        rows = [dict(clean_score=0.5, adversarial_score=0.5, linf=0.1)] * 3
        self.assertEqual(score_roc(rows)["auc"], 0.5)
        self.assertEqual(
            score_roc([dict(clean_score=0.1, adversarial_score=0.9, linf=0.1)])["auc"],
            1,
        )
        self.assertIsNone(
            score_roc([dict(clean_score=0.1, adversarial_score=0.1, linf=0)])["auc"]
        )
