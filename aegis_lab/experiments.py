"""实验计划展开与服务器运行。先打印任务，再按阶段运行同一份配置。"""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from ._core.training_config import TrainingConfig
from ._core.training_data import DATASETS
from .models import Classifier
from ._core.engine import load_model
from ._core.attacks import package_versions
from .features import default_layers
from .detection import Confidence, Entropy, Consistency, FeatureSqueezing, Mahalanobis, LayerRegression
from .anomaly import Energy, InputStatistics, OutputDrift
from .evaluation import evaluate, evaluate_detector, detection_metrics
from .ensembles import Ensemble
from .preprocessing import preprocess


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def expand_plan(plan):
    runs = []
    for dataset in plan["datasets"]:
        for seed in plan["seeds"]:
            for architecture in plan["architectures"]:
                methods = list(plan["main_methods"])
                if architecture in plan["distillation_architectures"]:
                    methods += ["distill_teacher", "distill"]
                for method in methods:
                    runs.append(make_run(plan, "main", dataset, architecture, method, 1, seed))
            for architecture in plan["long_tail"]["architectures"]:
                for ratio in plan["long_tail"]["imbalance_ratios"]:
                    for method in plan["long_tail"]["methods"]:
                        runs.append(make_run(plan, "long_tail", dataset, architecture, method, ratio, seed))
    return runs


def make_run(plan, group, dataset, architecture, method, ratio, seed):
    run_id = f"{dataset}-{architecture}-{method}-ir{ratio}-s{seed}"
    options = dict(plan["training"], dataset=dataset, architecture=architecture, method=method,
                   imbalance_ratio=ratio, seed=seed, data_dir=plan["data_dir"], output_dir=str(Path(plan["output_dir"]) / run_id))
    if method == "at_bsl":
        options["augmentation"] = "randaugment"
    if method == "distill":
        teacher_id = run_id.replace("-distill-", "-distill_teacher-")
        options["teacher_model_file"] = str(Path(plan["output_dir"]) / teacher_id / "model_registration.json")
    return dict(id=run_id, group=group, config=TrainingConfig(**options).model_dump())


def attack_grid(plan, smoke=False):
    if smoke:
        return [dict(method="fgsm", epsilon_255=8)]
    config = plan["evaluation"]
    return [dict(method=method, epsilon_255=budget, step_size_255=config["step_size_255"], restarts=config["restarts"])
            for method in config["attacks"] for budget in (config["autoattack_budgets_255"] if method == "autoattack" else config["budgets_255"])]


def load_run(run):
    folder = Path(run["config"]["output_dir"])
    spec = json.loads((folder / "model_registration.json").read_text(encoding="utf-8"))
    model, device = load_model(spec)
    return Classifier(model, spec["mean"], spec["std"]).to(device).eval(), spec


def make_loaders(plan, run, smoke=False):
    config = run["config"]
    dataset_class = DATASETS[config["dataset"]]["class"]
    transform = transforms.ToTensor()
    train = dataset_class(config["data_dir"], train=True, transform=transform, download=config["download"])
    test = dataset_class(config["data_dir"], train=False, transform=transform, download=config["download"])
    split = json.loads((Path(config["output_dir"]) / "split.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(config["seed"])
    # 每类取样确保 Mahalanobis 拟合覆盖所有类别。
    groups = [[] for _ in train.classes]
    for index in split["training_indices"]:
        groups[train.targets[index]].append(index)
    per_class = max(1, plan["detection"]["fit_samples"] // len(groups))
    fit_indices = [int(index) for group in groups for index in rng.permutation(group)[:per_class]]
    validation = rng.permutation(split["validation_indices"]).tolist()
    middle = len(validation) // 2
    development, calibration = validation[:middle], validation[middle:]
    limit = 8 if smoke else plan["evaluation"]["sample_limit"]
    test_indices = rng.permutation(len(test)).tolist()
    if limit:
        test_indices = test_indices[:limit]
    if smoke:
        fit_indices = [index for group in groups for index in group[:10]]
    write_json(Path(config["output_dir"]) / "evaluation_split.json", dict(fit_indices=fit_indices, development_indices=development, calibration_indices=calibration, test_indices=test_indices, seed=config["seed"]))
    options = dict(batch_size=8 if smoke else plan["evaluation"]["batch_size"], num_workers=0 if smoke else config["num_workers"], pin_memory=True)
    return {name: DataLoader(Subset(data, indices), shuffle=False, **options) for name, data, indices in
            [("fit", train, fit_indices), ("development", train, development), ("calibration", train, calibration), ("test", test, test_indices)]}


def run_training(run, download=False, smoke=False):
    config = run["config"]
    config["download"] = download
    if smoke:
        config.update(epochs=2, batch_size=8, num_workers=0, max_train_samples=100, max_validation_samples=200, attack_steps=1, validation_steps=1, taet_warmup_epochs=1)
    folder = Path(config["output_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    config_file = folder / "requested_config.json"
    write_json(config_file, config)
    command = [sys.executable, "-m", "aegis_lab._core.train", "--config", str(config_file)]
    if (folder / "last.pt").exists():
        command += ["--resume", str(folder / "last.pt")]
    subprocess.run(command, check=True)
    write_json(folder / "run.json", dict(**run, command=f"aegis train --config {config_file.as_posix()}", dependencies=package_versions()))


def run_evaluation(plan, run, model, loaders, smoke=False):
    folder = Path(run["config"]["output_dir"])
    cases = []
    def save_case(options, original, adversarial, label, clean_prediction, attacked_prediction):
        directory = folder / "cases"
        directory.mkdir(exist_ok=True)
        case_id = f"{options['method']}-{options['epsilon_255']}-{len(cases)}"
        paths = {}
        for name, tensor in [("original", original), ("adversarial", adversarial), ("difference", ((adversarial - original) * 8 + .5).clamp(0, 1))]:
            path = directory / f"{case_id}-{name}.png"
            transforms.ToPILImage()(tensor).save(path)
            paths[name] = str(path.resolve())
        cases.append(dict(attack=options["method"], epsilon_255=options["epsilon_255"], label=label, clean_prediction=clean_prediction, attacked_prediction=attacked_prediction, difference_scale=8, **paths))
    results = evaluate(model, loaders["test"], attack_grid(plan, smoke), case_callback=save_case)
    write_json(folder / "evaluation.json", dict(run_id=run["id"], results=results, cases=cases))


def detector_suite(plan, run, model):
    layers, target = default_layers(run["config"]["architecture"])
    methods = dict(confidence=lambda: Confidence(model), entropy=lambda: Entropy(model), consistency=lambda: Consistency(model),
                   feature_squeezing=lambda: FeatureSqueezing(model), mahalanobis=lambda: Mahalanobis(model, [target]),
                   layer_regression=lambda: LayerRegression(model, layers, target, hidden_dim=plan["detection"]["lr_hidden_dim"]))
    return {name: methods[name]() for name in plan["detection"]["methods"]}


def run_detection(plan, run, model, loaders, smoke=False):
    folder = Path(run["config"]["output_dir"])
    report = []
    for name, detector in detector_suite(plan, run, model).items():
        print(f"Detection: {run['id']} / {name}", flush=True)
        detector.fit(loaders["fit"], epochs=1 if smoke else plan["detection"]["lr_epochs"])
        detector.calibrate(loaders["calibration"], plan["detection"]["false_positive_rate"])
        detector.save(folder / "detectors" / f"{name}.pt")
        configurations = [item for item in attack_grid(plan, smoke) if item["epsilon_255"] == 8]
        rows = [evaluate_detector(detector, loaders["test"], item) for item in configurations]
        rows.append(evaluate_detector(detector, loaders["test"], dict(method="adaptive_pgd", epsilon_255=8, steps=1 if smoke else plan["detection"]["adaptive_steps"], step_size_255=2, seed=run["config"]["seed"]), adaptive=True))
        report.append(dict(method=name, options=detector.options, calibration=detector.calibration, results=rows, history=getattr(detector, "history", [])))
        write_json(folder / "detection.json", report)


def run_defenses(plan, run, model, loaders, all_runs, smoke=False):
    definitions = [(name, Ensemble([model], transforms=[dict(kind=name)], surrogate=True)) for name in plan["preprocessing"]]
    peers = [item for item in all_runs if item["group"] == "main" and item["config"]["method"] == "pgd_at"
             and item["config"]["dataset"] == run["config"]["dataset"] and item["config"]["seed"] == run["config"]["seed"]]
    members = [load_run(item)[0] for item in peers]
    definitions += [(name, Ensemble(members, method=name, surrogate=True)) for name in plan["ensembles"]]
    rows = []
    options = dict(method="fgsm" if smoke else "pgd100", epsilon_255=8, step_size_255=2)
    for name, defense in definitions:
        for mode in ("transfer", "adaptive"):
            # 用代理生成攻击，再用实际变换/投票计算分类结果。
            from .attacks import attack
            from .evaluation import classification_metrics
            labels, clean_predictions, attacked_predictions = [], [], []
            for images, targets in loaders["test"]:
                images, targets = images.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)
                defense.surrogate = True
                adversarial = attack(model if mode == "transfer" else defense, images, targets, **options)
                defense.surrogate = False
                with torch.no_grad():
                    clean_logits, attacked_logits = defense(images), defense(adversarial)
                labels.extend(targets.tolist())
                clean_predictions.extend(clean_logits.argmax(1).tolist())
                attacked_predictions.extend(attacked_logits.argmax(1).tolist())
            rows.append(dict(method=name, mode=mode, parameters=options, members=[item["id"] for item in peers] if name in plan["ensembles"] else [run["id"]], **classification_metrics(labels, clean_predictions, attacked_predictions, clean_logits.shape[1])))
    write_json(Path(run["config"]["output_dir"]) / "defenses.json", rows)


def run_anomaly(plan, run, model, loaders, smoke=False):
    device = next(model.parameters()).device
    _, target = default_layers(run["config"]["architecture"])
    detectors = dict(input_statistics=InputStatistics(model), energy=Energy(model), mahalanobis=Mahalanobis(model, [target]))
    config = run["config"]
    other = "cifar100" if config["dataset"] == "cifar10" else "cifar10"
    ood_data = DATASETS[other]["class"](config["data_dir"], train=False, download=config["download"], transform=transforms.ToTensor())
    count = len(loaders["test"].dataset)
    indices = np.random.default_rng(config["seed"]).permutation(len(ood_data))[:count].tolist()
    ood_loader = DataLoader(Subset(ood_data, indices), batch_size=loaders["test"].batch_size)
    rows = []
    for name in plan["anomaly"]["methods"]:
        detector = detectors[name].fit(loaders["fit"]).calibrate(loaders["calibration"], plan["detection"]["false_positive_rate"])
        folder = Path(config["output_dir"]) / "anomaly"
        folder.mkdir(exist_ok=True)
        detector.save(folder / f"{name}.pt")
        scores = {kind: [] for kind in ["clean", "ood"] + plan["anomaly"]["corruptions"]}
        with torch.no_grad():
            for images, _ in loaders["test"]:
                images = images.to(device)
                scores["clean"].extend(detector.score(images).tolist())
                for kind in plan["anomaly"]["corruptions"]:
                    altered = (images + 0.25).clamp(0, 1) if kind == "brightness" else preprocess(images, [dict(kind=kind, noise_std=0.1, sigma=1.0)])
                    scores[kind].extend(detector.score(altered).tolist())
            for images, _ in ood_loader:
                scores["ood"].extend(detector.score(images.to(device)).tolist())
        rows.extend(dict(method=name, condition=condition, ood_dataset=other if condition == "ood" else None,
                         **detection_metrics(scores["clean"], values, detector.threshold)) for condition, values in scores.items() if condition != "clean")
    def probabilities(loader):
        with torch.no_grad():
            return torch.cat([model(images.to(device)).softmax(1).cpu() for images, _ in loader])
    calibration = probabilities(loaders["calibration"])
    window = min(4 if smoke else plan["anomaly"]["window_size"], len(calibration) // 2)
    monitor = OutputDrift(window).fit(probabilities(loaders["development"])).calibrate(calibration)
    monitor.save(Path(config["output_dir"]) / "anomaly" / "output_drift.pt")
    events = []
    for condition, loader in [("clean", loaders["test"]), ("ood", ood_loader)]:
        monitor.pending = []
        events.extend(dict(condition=condition, **event) for event in monitor.update(probabilities(loader)))
    write_json(Path(config["output_dir"]) / "anomaly.json", dict(results=rows, drift=events, window_size=window, ood_dataset=other))


def execute_plan(plan, *, stage="all", group="all", only="", download=False, smoke=False):
    if smoke:
        plan = dict(plan, output_dir=plan["output_dir"] + "-smoke")
    runs = expand_plan(plan)
    selected = [item for item in runs if (group == "all" or item["group"] == group) and only in item["id"]]
    if stage == "plan":
        return dict(name=plan["name"], training_jobs=len(selected), main_jobs=sum(item["group"] == "main" for item in selected),
                    long_tail_jobs=sum(item["group"] == "long_tail" for item in selected), attack_settings=len(attack_grid(plan, smoke)), runs=selected)
    write_json(Path(plan["output_dir"]) / "plan.json", plan)
    if stage in ("all", "train"):
        for run in selected:
            run_training(run, download, smoke)
    if stage in ("all", "evaluate", "detect", "defend", "anomaly"):
        for run in selected:
            torch.manual_seed(run["config"]["seed"])
            folder = Path(run["config"]["output_dir"])
            run["config"] = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            model, spec = load_run(run)
            loaders = make_loaders(plan, run, smoke)
            if stage in ("all", "evaluate"):
                run_evaluation(plan, run, model, loaders, smoke)
            chosen = run["group"] == "main" and run["config"]["architecture"] in plan["detection"]["architectures"] and run["config"]["method"] in plan["detection"]["training_methods"]
            if chosen and stage in ("all", "detect"):
                run_detection(plan, run, model, loaders, smoke)
            if chosen and stage in ("all", "anomaly"):
                run_anomaly(plan, run, model, loaders, smoke)
            if chosen and run["config"]["method"] == "pgd_at" and stage in ("all", "defend"):
                run_defenses(plan, run, model, loaders, runs, smoke)
            del model
            torch.cuda.empty_cache()
    return dict(completed_stage=stage, selected_runs=len(selected), output_dir=plan["output_dir"])
