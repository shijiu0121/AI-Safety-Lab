"""公开库的数值、梯度、保存恢复和静态成果导出测试。"""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from aegis_lab import Classifier, attack
from aegis_lab.detection import Confidence, Entropy, Consistency, FeatureSqueezing, Mahalanobis, LayerRegression, load_detector
from aegis_lab.anomaly import Energy, InputStatistics, OutputDrift
from aegis_lab.evaluation import classification_metrics, detection_metrics, evaluate, adaptive_attack
from aegis_lab.ensembles import Ensemble
from aegis_lab.experiments import expand_plan, attack_grid
from aegis_lab.publishing import export_site


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, images):
        return self.fc(self.pool(self.features(images)).flatten(1))


class LibraryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = Classifier(SmallNet()).eval()
        self.images = torch.rand(24, 3, 8, 8)
        self.labels = torch.arange(24) % 2
        self.loader = DataLoader(TensorDataset(self.images, self.labels), batch_size=6)
        with torch.no_grad():
            predicted = self.model(self.images).argmax(1)
        self.correct_loader = DataLoader(TensorDataset(self.images, predicted), batch_size=6)

    def test_attack_preserves_batchnorm_modes_and_budget(self):
        self.model.train()
        self.model.model.features.eval()
        before = [m.training for m in self.model.modules()]
        result = attack(self.model, self.images, self.labels, "pgd", steps=2, epsilon_255=4)
        self.assertLessEqual(float((result-self.images).abs().max()), 4/255+1e-6)
        self.assertEqual(before, [m.training for m in self.model.modules()])
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_feature_squeezing_formula_and_input_gradient(self):
        from aegis_lab.preprocessing import preprocess
        detector = FeatureSqueezing(self.model)
        images = self.images.requires_grad_(True)
        probabilities = self.model(images).softmax(1)
        expected = torch.stack([(probabilities-self.model(preprocess(images,[options],bpda=True)).softmax(1)).abs().sum(1) for options in [dict(kind="bit_depth",bits=5),dict(kind="median",kernel=3)]]).max(0).values
        self.assertTrue(torch.allclose(detector.score(images), expected))
        gradient = torch.autograd.grad(detector.score(images).sum(), images)[0]
        self.assertTrue(torch.isfinite(gradient).all())

    def test_detectors_calibrate_and_roundtrip(self):
        detectors = [Confidence(self.model), Entropy(self.model), Consistency(self.model), FeatureSqueezing(self.model),
                     Mahalanobis(self.model,["model.pool"]), LayerRegression(self.model,["model.features"],"model.pool",hidden_dim=8),
                     Energy(self.model), InputStatistics(self.model)]
        with tempfile.TemporaryDirectory() as folder:
            for index, detector in enumerate(detectors):
                fit_loader = self.correct_loader if isinstance(detector, LayerRegression) else self.loader
                detector.fit(fit_loader, epochs=2).calibrate(self.loader)
                before = detector.score(self.images).detach()
                path = Path(folder)/f"{index}.pt"
                detector.save(path)
                restored = load_detector(path,self.model)
                self.assertTrue(torch.allclose(before,restored.score(self.images)))
                self.assertEqual(detector.threshold,restored.threshold)
                self.assertLessEqual(float(detector.predict(self.images).float().mean()),.05)

    def test_mahalanobis_matches_numpy_covariance(self):
        detector = Mahalanobis(self.model,["model.pool"],regularization=1e-4).fit(self.loader)
        from aegis_lab.features import activations, pooled
        features = pooled(activations(self.model,self.images,["model.pool"])[1][0]).detach().numpy()
        means = np.stack([features[self.labels.numpy()==category].mean(0) for category in range(2)])
        residual = features-means[self.labels.numpy()]
        covariance = residual.T@residual/len(features)+np.eye(4)*1e-4
        expected = np.stack([.5*np.einsum('ni,ij,nj->n',features-mean,np.linalg.inv(covariance),features-mean) for mean in means],axis=1).min(1)
        np.testing.assert_allclose(detector.score(self.images).detach().numpy(),expected,rtol=2e-3,atol=1e-3)

    def test_layer_regression_optimizes_feature_prediction(self):
        detector = LayerRegression(self.model,["model.features"],"model.pool",hidden_dim=8).fit(self.correct_loader,epochs=12,learning_rate=.01)
        self.assertLess(detector.history[-1]["loss"],detector.history[0]["loss"])
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_output_drift_buffers_windows_and_restores(self):
        reference = torch.tensor([[.6,.4]]*16)
        monitor = OutputDrift(4).fit(reference).calibrate(reference)
        self.assertEqual(monitor.update(reference[:3]),[])
        events = monitor.update(torch.tensor([[.01,.99]]*5))
        self.assertEqual(len(events),2)
        self.assertTrue(events[-1]["flagged"])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"drift.pt"
            monitor.save(path)
            restored=OutputDrift.load(path)
            self.assertEqual(restored.window_size,4)
            self.assertEqual(restored.threshold,monitor.threshold)

    def test_metrics_and_tied_roc(self):
        report = detection_metrics([.2,.2],[.2,.2],.2)
        self.assertEqual(report["auroc"],.5)
        self.assertEqual(report["fpr"],0)
        report = classification_metrics([0,0,1],[0,0,0],[0,1,0],2)
        self.assertAlmostEqual(report["attack_success_rate"],.5)
        self.assertAlmostEqual(report["balanced_robust_accuracy"],.25)

    def test_adaptive_attack_and_ensemble_evaluate(self):
        detector=Confidence(self.model).calibrate(self.loader)
        adversarial=adaptive_attack(self.model,detector,self.images,self.labels,steps=2)
        self.assertLessEqual(float((adversarial-self.images).abs().max()),8/255+1e-6)
        for method in ["soft_vote","hard_vote","logit_mean","random_member"]:
            model=Ensemble([self.model,self.model],method=method,surrogate=True).eval()
            report=evaluate(model,self.loader,[dict(method="fgsm",epsilon_255=2)])
            self.assertEqual(report[0]["samples"],24)

    def test_expanded_plan_has_teacher_dependencies_and_all_groups(self):
        path=Path(__file__).resolve().parents[1]/"configs/experiment-plan.json"
        plan=json.loads(path.read_text(encoding="utf-8"))
        runs=expand_plan(plan)
        self.assertEqual(len(runs),42)
        self.assertEqual(len(attack_grid(plan)),16)
        seen=set()
        for run in runs:
            if run["config"]["method"]=="distill":
                self.assertIn(run["id"].replace("-distill-","-distill_teacher-"),seen)
            seen.add(run["id"])

    def test_export_creates_downloadable_static_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/"runs"; run=source/"one"; run.mkdir(parents=True)
            (run/"run.json").write_text(json.dumps(dict(id="one",group="main",config=dict(dataset="cifar10",architecture="tiny",method="erm"),command="aegis train --config c.json",dependencies={})),encoding="utf-8")
            (run/"config.json").write_text('{}',encoding="utf-8")
            (run/"metrics.jsonl").write_text('{"epoch":1}\n',encoding="utf-8")
            torch.save({},run/"best_weights.pt")
            result=export_site(source,root/"site",include_weights=True)
            self.assertEqual(result["experiments"],1)
            manifest=json.loads((root/"site/results.json").read_text(encoding="utf-8"))
            for url in manifest["experiments"][0]["downloads"].values():
                self.assertTrue((root/"site"/url).is_file())
            self.assertNotIn(str(root),manifest["experiments"][0]["downloads"]["weights"])


if __name__ == "__main__":
    unittest.main()
