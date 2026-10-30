"""防御评测和独立验证集阈值校准。分类鲁棒性、拒绝率与检测指标分别报告。"""

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field, model_validator
from torch.utils.data import DataLoader, Subset

from . import engine
from .attacks import run_attack, margin_loss, package_versions
from .attack_protocol import AttackConfig
from .batch_engine import (
    BatchConfig,
    attack_grid,
    digest_json,
    save_json,
    write_csv,
    summarize_rows,
)
from .data_sources import inspect_dataset, load_dataset, check_dataset_model
from .defenses import (
    DefenseRuntime,
    DefenseConfig,
    AttackSurrogate,
    identity_spec,
    transform_images,
    TransformConfig,
)


class DefenseEvaluationConfig(BatchConfig):
    model_ids: list[str] = Field(default=[], max_length=1)
    defense_ids: list[str] = Field(min_length=1, max_length=8)
    modes: list[Literal["transfer", "adaptive"]] = Field(
        default=["transfer", "adaptive"], min_length=1, max_length=2
    )
    attacks: list[
        Literal[
            "fgsm",
            "pgd20",
            "pgd100",
            "cw_linf",
            "autoattack",
            "deepfool_linf",
            "deepfool_l2",
        ]
    ] = Field(default=["pgd20", "cw_linf"], min_length=1, max_length=7)
    evade_detector: bool = True
    evasion_weight: float = Field(default=1, ge=0, le=100)

    @model_validator(mode="after")
    def distinct(self):
        if len(set(self.defense_ids)) != len(self.defense_ids) or len(
            set(self.modes)
        ) != len(self.modes):
            raise ValueError("防御和威胁模式不能重复。")
        if self.model_ids:
            raise ValueError(
                "防御评测通过 defense_ids 指定成员，不另外接受 model_ids。"
            )
        return self


def detector_penalty(runtime, images, logits):
    config = runtime.config.detector
    probabilities = logits.softmax(1)
    if config.method == "confidence":
        score = 1 - probabilities.max(1).values
    elif config.method == "entropy":
        score = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1) / np.log(
            probabilities.shape[1]
        )
    elif config.method == "consistency":
        filtered = transform_images(images, [TransformConfig(kind="median")], bpda=True)
        other = runtime.attack_logits(filtered).softmax(1)
        score = 0.5 * (probabilities - other).abs().sum(1)
    else:
        score = probabilities[:, 0] * 0
    penalty = F.relu(score - config.threshold) if config.method != "none" else score
    flat = images.flatten(1)
    penalty = penalty + F.relu(config.min_input_std - flat.std(1, unbiased=False))
    soft_saturation = (
        torch.sigmoid((1 / 255 - flat) * 100) + torch.sigmoid((flat - 254 / 255) * 100)
    ).mean(1)
    return penalty + F.relu(soft_saturation - config.max_saturation)


def adaptive_projected_attack(
    runtime, images, labels, options, evade, weight, cancelled=None
):
    """BPDA/EOT 梯度攻击；检测规避为显式惩罚项，最终按真实预测/检测判据选样本。"""
    protocol = options.protocol()
    runtime.attack_calls = 0
    runtime.attack_seed = options.seed
    if protocol["budget"] == 0:
        return images.clone(), dict(protocol=protocol)
    generator = torch.Generator(device=images.device).manual_seed(options.seed)
    best = images.clone()
    best_score = torch.full_like(labels, -torch.inf, dtype=images.dtype)
    best_success = torch.zeros_like(labels, dtype=torch.bool)
    for restart in range(1 if options.algorithm == "fgsm" else options.restarts):
        current = (
            images.clone()
            if options.algorithm == "fgsm"
            else (
                images
                + (
                    torch.rand(images.shape, device=images.device, generator=generator)
                    * 2
                    - 1
                )
                * protocol["budget"]
            ).clamp(0, 1)
        )
        for _ in range(protocol["iterations"]):
            if cancelled and cancelled():
                raise engine.Cancelled()
            current = current.detach().requires_grad_(True)
            logits = runtime.attack_logits(current)
            objective = (
                margin_loss(logits, labels, options.cw_confidence)
                if options.algorithm == "cw_linf"
                else F.cross_entropy(logits, labels, reduction="none")
            )
            if evade:
                objective = objective - weight * detector_penalty(
                    runtime, current, logits
                )
            gradient = torch.autograd.grad(objective.mean(), current)[0]
            if not torch.isfinite(gradient).all():
                raise ValueError("自适应攻击梯度含有 NaN/Inf。")
            alpha = (
                protocol["budget"]
                if options.algorithm == "fgsm"
                else protocol["step_size"]
            )
            current = (
                (current.detach() + alpha * gradient.sign())
                .clamp(images - protocol["budget"], images + protocol["budget"])
                .clamp(0, 1)
            )
            measured = runtime.measure(current, options.seed + 3000000)
            success = measured["prediction"].ne(labels)
            if evade:
                success &= ~measured["flagged"]
            with torch.no_grad():
                score = F.cross_entropy(measured["logits"], labels, reduction="none")
                if evade:
                    score = score - weight * F.relu(
                        measured["score"] - runtime.config.detector.threshold
                    )
                keep = (success & ~best_success) | (
                    (success == best_success) & (score > best_score)
                )
                best = torch.where(keep[:, None, None, None], current, best)
                best_score = torch.where(keep, score, best_score)
                best_success |= success
    protocol.update(
        adaptive=True, evade_detector=evade, evasion_weight=weight, **runtime.protocol()
    )
    return best.detach(), dict(protocol=protocol)


def selected_samples(count, limit, seed):
    if limit and limit < count:
        return torch.randperm(count, generator=torch.Generator().manual_seed(seed))[
            :limit
        ].tolist()
    return list(range(count))


def load_resources(record):
    from . import store

    return [store.get("models", key) for key in record["model_ids"]]


def calibration_signature(profile, specs):
    return digest_json(
        dict(
            config=DefenseConfig.model_validate(profile).model_dump(),
            models=[
                dict(
                    id=item["id"],
                    weights=engine.sha256(item["weights_path"]),
                    loader=(
                        engine.sha256(item["loader_path"])
                        if item["architecture"] == "custom"
                        else None
                    ),
                    contract={
                        key: item[key]
                        for key in (
                            "num_classes",
                            "input_size",
                            "categories",
                            "mean",
                            "std",
                        )
                    },
                )
                for item in specs
            ],
        )
    )


def calibrate(
    profile,
    specs,
    dataset,
    directory,
    target_fpr=0.05,
    sample_limit=0,
    batch_size=32,
    cancelled=None,
    progress=None,
):
    if dataset["split"] != "validation":
        raise ValueError("阈值只能在独立 validation 划分上校准，不接受测试集。")
    config = DefenseConfig.model_validate(profile["config"])
    if config.detector.method == "none":
        raise ValueError("请先选择检测方法。")
    info = inspect_dataset(dataset)
    check_dataset_model(dict(dataset, inspection=info), specs[0])
    runtime = DefenseRuntime(config, specs)
    indices = selected_samples(info["sample_count"], sample_limit, config.seed)
    loader = DataLoader(
        Subset(load_dataset(dataset, specs[0]["input_size"]), indices),
        batch_size=batch_size,
    )
    scores = []
    input_digests = []
    for index, (images, labels) in enumerate(loader):
        if cancelled and cancelled():
            raise engine.Cancelled()
        input_digests.append(
            hashlib.sha256(
                images.numpy().tobytes() + labels.numpy().tobytes()
            ).hexdigest()
        )
        measured = runtime.measure(images.to(runtime.device), config.seed + index)
        scores.extend(measured["score"].cpu().tolist())
        if progress:
            progress(round(len(scores) / len(indices) * 99), "校准验证集检测分数", None)
    # 使用 higher 分位数和严格 > 判据，经验误报数不会因插值被额外放大。
    threshold = float(np.quantile(scores, 1 - target_fpr, method="higher"))
    result = dict(
        threshold=threshold,
        requested_fpr=target_fpr,
        empirical_fpr=float(np.mean(np.asarray(scores) > threshold)),
        sample_count=len(scores),
        dataset_id=dataset["id"],
        dataset_split=dataset["split"],
        dataset_fingerprint=info["source_fingerprint"],
        input_sha256=input_digests,
        sample_indices=indices,
        profile_signature=calibration_signature(profile["config"], specs),
        note="使用验证集分数分位数校准检测阈值。",
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "calibration.json", result)
    write_csv(
        directory / "scores.csv",
        [dict(sample_index=indices[i], score=score) for i, score in enumerate(scores)],
    )
    return result


def detection_metrics(rows):
    count = len(rows)
    accepted_clean = [row for row in rows if not row["clean_flagged"]]
    accepted_adv = [row for row in rows if not row["adversarial_flagged"]]
    perturbed = [row for row in rows if row["linf"] > 1e-12]
    successful = [
        row
        for row in perturbed
        if row["clean_correct"] and row["adversarial_prediction"] != row["label"]
    ]
    return dict(
        clean_flag_rate=sum(row["clean_flagged"] for row in rows) / count,
        attacked_flag_rate=(
            sum(row["adversarial_flagged"] for row in perturbed) / len(perturbed)
            if perturbed
            else None
        ),
        detected_successful_attacks=(
            sum(row["adversarial_flagged"] for row in successful) / len(successful)
            if successful
            else None
        ),
        clean_coverage=len(accepted_clean) / count,
        adversarial_coverage=len(accepted_adv) / count,
        accepted_clean_accuracy=(
            sum(row["clean_correct"] for row in accepted_clean) / len(accepted_clean)
            if accepted_clean
            else None
        ),
        accepted_adversarial_accuracy=(
            sum(row["adversarial_prediction"] == row["label"] for row in accepted_adv)
            / len(accepted_adv)
            if accepted_adv
            else None
        ),
        accepted_correct_rate=sum(
            row["adversarial_prediction"] == row["label"]
            and not row["adversarial_flagged"]
            for row in rows
        )
        / count,
        undetected_error_rate=sum(
            row["adversarial_prediction"] != row["label"]
            and not row["adversarial_flagged"]
            for row in rows
        )
        / count,
        perturbed_count=len(perturbed),
    )


def score_roc(rows):
    positives = [row["adversarial_score"] for row in rows if row["linf"] > 1e-12]
    negatives = [row["clean_score"] for row in rows]
    if not positives or not negatives:
        return dict(auc=None, points=[], note="零扰动时没有正例，不计算 ROC。")
    scores = np.asarray(positives + negatives)
    labels = np.asarray([1] * len(positives) + [0] * len(negatives))
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tpr = np.r_[0, np.cumsum(sorted_labels)[ends] / len(positives)]
    fpr = np.r_[0, np.cumsum(1 - sorted_labels)[ends] / len(negatives)]
    auc = float(sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    selected = np.unique(np.linspace(0, len(tpr) - 1, min(200, len(tpr))).astype(int))
    return dict(
        auc=auc,
        points=[dict(fpr=float(fpr[i]), tpr=float(tpr[i])) for i in selected],
        positive_count=len(positives),
        negative_count=len(negatives),
        note="ROC 使用原图与非零扰动图的检测分数，AUC 使用完整排序计算。",
    )


def run_defense_suite(
    config, dataset, profiles, members, directory, progress=None, cancelled=None
):
    config = (
        config
        if isinstance(config, DefenseEvaluationConfig)
        else DefenseEvaluationConfig.model_validate(config)
    )
    if [item["id"] for item in profiles] != config.defense_ids:
        raise ValueError("防御定义与 defense_ids 不一致。")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inspection = inspect_dataset(dataset)
    if (
        dataset.get("status") != "validated"
        or inspection["source_fingerprint"]
        != dataset["inspection"]["source_fingerprint"]
    ):
        raise ValueError("数据集未检查或已改变。")
    snapshots = []
    for profile in profiles:
        specs = members[profile["id"]]
        defense_options = DefenseConfig.model_validate(profile["config"])
        constrained = defense_options.aggregation in (
            "hard_vote",
            "random_member",
        ) or any(
            item.kind in ("bit_depth", "median", "noise")
            for item in defense_options.transforms
        )
        if (
            "adaptive" in config.modes
            and any(
                name not in ("fgsm", "pgd20", "pgd100", "cw_linf")
                for name in config.attacks
            )
            and (constrained or config.evade_detector)
        ):
            raise ValueError(
                "当前自适应防御协议使用 FGSM/PGD-20/PGD-100/CW；AA/DeepFool 可在迁移评测或确定性可微分类协议下运行。"
            )
        for spec in specs:
            check_dataset_model(dataset, spec)
        signature = calibration_signature(profile["config"], specs)
        calibration = profile.get("calibration")
        if calibration and calibration["applied_signature"] != signature:
            raise ValueError(
                "防御配置或成员与阈值校准时不同，请重新校准或显式清除校准。"
            )
        snapshots.append(
            dict(profile=profile, models=specs, resource_signature=signature)
        )
    manifest = dict(
        config=config.model_dump(),
        dataset=dataset,
        profiles=snapshots,
        dependencies=package_versions(),
        code={
            name: engine.sha256(Path(__file__).parent / name)
            for name in [
                "defenses.py",
                "defense_evaluation.py",
                "attacks.py",
                "engine.py",
                "data_sources.py",
            ]
        },
    )
    signature = digest_json(manifest)
    manifest_path = directory / "manifest.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8"))["signature"]
        != signature
    ):
        raise ValueError("恢复配置、资源或代码不同，请新建实验。")
    save_json(manifest_path, dict(signature=signature, **manifest))
    indices = selected_samples(
        inspection["sample_count"], config.sample_limit, config.seed
    )
    save_json(directory / "sample_indices.json", indices)
    report = dict(
        kind="defense_suite",
        signature=signature,
        dataset_name=dataset["name"],
        dataset_split=dataset["split"],
        sample_count=len(indices),
        subset=len(indices) < inspection["sample_count"],
        classes=inspection["classes"],
        cells=[],
        combined=[],
        note="按防御配置、攻击方法、扰动预算和攻击模式汇总结果。",
    )
    all_rows = []
    grid = attack_grid(config)
    total = (
        len(profiles)
        * len(grid)
        * len(config.modes)
        * ((len(indices) + config.batch_size - 1) // config.batch_size)
    )
    done = 0
    combined = {}
    for profile in profiles:
        specs = members[profile["id"]]
        runtime = DefenseRuntime(profile["config"], specs)
        loader = DataLoader(
            Subset(load_dataset(dataset, specs[0]["input_size"]), indices),
            batch_size=config.batch_size,
        )
        for attack in grid:
            for mode in config.modes:
                projected = attack.algorithm in ("fgsm", "pgd20", "pgd100", "cw_linf")
                if (
                    mode == "adaptive"
                    and not projected
                    and (
                        runtime.stochastic
                        or runtime.bpda
                        or runtime.config.aggregation == "hard_vote"
                        or config.evade_detector
                    )
                ):
                    raise ValueError(
                        "该防御的自适应评测请用 FGSM/PGD-20/PGD-100/CW；不会将删减或代理套件标成 AA standard。"
                    )
                cell_id = digest_json(
                    dict(defense=profile["id"], attack=attack.model_dump(), mode=mode)
                )[:16]
                folder = directory / cell_id
                folder.mkdir(exist_ok=True)
                rows = []
                examples = []
                elapsed = 0
                reused = 0
                for batch_index, (images, labels) in enumerate(loader):
                    if cancelled and cancelled():
                        raise engine.Cancelled()
                    input_hash = hashlib.sha256(
                        images.numpy().tobytes() + labels.numpy().tobytes()
                    ).hexdigest()
                    path = folder / f"batch_{batch_index:06d}.json"
                    if path.exists():
                        saved = json.loads(path.read_text(encoding="utf-8"))
                        if saved["input_sha256"] != input_hash:
                            raise ValueError("缓存批次输入改变，无法恢复。")
                        reused += 1
                    else:
                        images, labels = images.to(runtime.device), labels.to(
                            runtime.device
                        )
                        if runtime.device.type == "cuda":
                            torch.cuda.synchronize(runtime.device)
                        started = time.perf_counter()
                        seeded = attack.model_copy(
                            update=dict(seed=config.seed + batch_index)
                        )
                        base = runtime.models[0]
                        with torch.no_grad():
                            base_clean = engine.forward(base, images, specs[0]).argmax(
                                1
                            )
                        clean = runtime.measure(images, seeded.seed + 3000000)
                        # 基准是成员列表第一个模型，对其单独攻击；不是把迁移攻击误称白盒攻击。
                        base_adv, _ = run_attack(
                            base,
                            images,
                            labels,
                            specs[0],
                            seeded,
                            cancelled=cancelled,
                            log_path=(
                                folder / f"baseline_aa_{batch_index}.log"
                                if attack.algorithm == "autoattack"
                                else None
                            ),
                        )
                        with torch.no_grad():
                            base_pred = engine.forward(base, base_adv, specs[0]).argmax(
                                1
                            )
                        if mode == "transfer":
                            adversarial = base_adv
                            details = dict(
                                protocol=dict(
                                    seeded.protocol(),
                                    attack_source="first member model; transfer to defense",
                                )
                            )
                        elif projected:
                            adversarial, details = adaptive_projected_attack(
                                runtime,
                                images,
                                labels,
                                seeded,
                                config.evade_detector,
                                config.evasion_weight,
                                cancelled,
                            )
                        else:
                            runtime.attack_calls = 0
                            runtime.attack_seed = seeded.seed
                            adversarial, details = run_attack(
                                AttackSurrogate(runtime),
                                images,
                                labels,
                                identity_spec(specs[0]),
                                seeded,
                                cancelled=cancelled,
                                log_path=(
                                    folder / f"adaptive_aa_{batch_index}.log"
                                    if attack.algorithm == "autoattack"
                                    else None
                                ),
                            )
                        measured = runtime.measure(adversarial, seeded.seed + 3000000)
                        if runtime.device.type == "cuda":
                            torch.cuda.synchronize(runtime.device)
                        batch_rows = []
                        batch_examples = []
                        for local in range(len(labels)):
                            label = int(labels[local])
                            prediction = int(measured["prediction"][local])
                            clean_prediction = int(clean["prediction"][local])
                            delta = (adversarial[local] - images[local]).flatten()
                            row = dict(
                                sample_index=indices[
                                    batch_index * config.batch_size + local
                                ],
                                label=label,
                                clean_prediction=clean_prediction,
                                adversarial_prediction=prediction,
                                clean_correct=clean_prediction == label,
                                robust_correct=clean_prediction == label
                                and prediction == label,
                                base_clean_correct=int(base_clean[local]) == label,
                                base_adversarial_correct=int(base_pred[local]) == label,
                                clean_flagged=bool(clean["flagged"][local]),
                                adversarial_flagged=bool(measured["flagged"][local]),
                                clean_score=float(clean["score"][local]),
                                adversarial_score=float(measured["score"][local]),
                                linf=float(delta.abs().max()),
                                l2=float(delta.norm()),
                            )
                            batch_rows.append(row)
                            if (
                                clean_prediction == label
                                and prediction != label
                                and len(examples) + len(batch_examples) < 3
                            ):
                                prefix = f'case_{row["sample_index"]}'
                                processed = transform_images(
                                    adversarial[local : local + 1],
                                    runtime.config.transforms,
                                    torch.Generator(device=runtime.device).manual_seed(
                                        seeded.seed
                                    ),
                                )
                                for kind, tensor in [
                                    ("original", images[local : local + 1]),
                                    ("adversarial", adversarial[local : local + 1]),
                                    ("processed", processed),
                                ]:
                                    engine.save_tensor(
                                        tensor, folder / f"{prefix}_{kind}.png"
                                    )
                                batch_examples.append(
                                    dict(
                                        sample_index=row["sample_index"],
                                        original=f"{cell_id}/{prefix}_original.png",
                                        adversarial=f"{cell_id}/{prefix}_adversarial.png",
                                        processed=f"{cell_id}/{prefix}_processed.png",
                                        note="处理图为一次变换预览；随机/集成预测可能使用多次采样。",
                                    )
                                )
                        saved = dict(
                            input_sha256=input_hash,
                            rows=batch_rows,
                            seconds=time.perf_counter() - started,
                            examples=batch_examples,
                            attack_protocol=details["protocol"],
                        )
                        save_json(path, saved)
                    rows.extend(saved["rows"])
                    elapsed += saved["seconds"]
                    examples.extend(saved["examples"])
                    done += 1
                    if progress:
                        progress(
                            round(done / total * 99),
                            f'{profile["name"]} / {attack.algorithm} / {mode} / 批次 {batch_index+1}',
                            report,
                        )
                metrics = summarize_rows(rows, specs[0]["num_classes"])
                base_clean = sum(row["base_clean_correct"] for row in rows) / len(rows)
                base_robust = sum(
                    row["base_clean_correct"] and row["base_adversarial_correct"]
                    for row in rows
                ) / len(rows)
                cell = dict(
                    id=cell_id,
                    defense_id=profile["id"],
                    model_id=profile["id"],
                    model_name=profile["name"],
                    defense_name=profile["name"],
                    mode=mode,
                    protocol=dict(
                        attack.protocol(), name=attack.protocol()["name"] + " / " + mode
                    ),
                    defense_protocol=runtime.protocol(),
                    actual_attack_protocol=saved["attack_protocol"],
                    metrics=metrics,
                    detection=detection_metrics(rows),
                    baseline=dict(
                        model_name=specs[0]["name"],
                        clean_accuracy=base_clean,
                        robust_accuracy=base_robust,
                    ),
                    seconds=round(elapsed, 4),
                    reused_batches=reused,
                    examples=examples,
                )
                cell["roc"] = score_roc(rows)
                report["cells"].append(cell)
                write_csv(folder / "samples.csv", rows)
                all_rows.extend(
                    dict(
                        defense=profile["name"],
                        mode=mode,
                        attack=attack.algorithm,
                        norm=attack.protocol()["norm"],
                        budget=attack.protocol()["budget"],
                        **row,
                    )
                    for row in rows
                )
                key = (
                    profile["id"],
                    mode,
                    attack.protocol()["norm"],
                    attack.protocol()["budget"],
                )
                if key not in combined:
                    combined[key] = dict(
                        rows={row["sample_index"]: dict(row) for row in rows},
                        attacks=[],
                    )
                else:
                    for row in rows:
                        existing = combined[key]["rows"][row["sample_index"]]
                        if existing["clean_prediction"] != row["clean_prediction"]:
                            raise ValueError("不同攻击正常预测不一致，拒绝合并。")
                        existing["robust_correct"] &= row["robust_correct"]
                combined[key]["attacks"].append(attack.algorithm)
                save_json(directory / "report.json", report)
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for key, value in combined.items():
        report["combined"].append(
            dict(
                defense_id=key[0],
                mode=key[1],
                norm=key[2],
                budget=key[3],
                attacks=value["attacks"],
                metrics=summarize_rows(
                    list(value["rows"].values()), len(inspection["classes"])
                ),
            )
        )
    save_json(directory / "report.json", report)
    write_csv(directory / "samples.csv", all_rows)
    write_csv(
        directory / "summary.csv",
        [
            dict(
                defense=cell["defense_name"],
                mode=cell["mode"],
                attack=cell["protocol"]["name"],
                norm=cell["protocol"]["norm"],
                budget=cell["protocol"]["budget"],
                clean_accuracy=cell["metrics"]["clean_accuracy"],
                robust_accuracy=cell["metrics"]["robust_accuracy"],
                baseline_robust_accuracy=cell["baseline"]["robust_accuracy"],
                **cell["detection"],
                seconds=cell["seconds"],
            )
            for cell in report["cells"]
        ],
    )
    write_csv(
        directory / "per_class.csv",
        [
            dict(
                defense=cell["defense_name"],
                mode=cell["mode"],
                attack=cell["protocol"]["name"],
                budget=cell["protocol"]["budget"],
                class_name=inspection["classes"][item["class_index"]],
                **item,
            )
            for cell in report["cells"]
            for item in cell["metrics"]["per_class"]
        ],
    )
    from .reporting import write_html_report

    write_html_report(report, directory / "report.html")
    return report
