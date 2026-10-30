"""多模型、多攻击、多预算的批量实验。以已完成批次为恢复单位。"""

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader, Subset

from . import engine
from .attack_protocol import (
    AttackConfig,
    AttackName,
    CATALOG,
    check_attack_compatibility,
)
from .attacks import package_versions, run_attack
from .data_sources import check_dataset_model, inspect_dataset, load_dataset


class BatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    dataset_id: str
    model_ids: list[str] = Field(min_length=1, max_length=12)
    attacks: list[AttackName] = Field(
        default=["pgd20", "pgd100", "cw_linf", "autoattack", "deepfool_linf"],
        min_length=1,
        max_length=8,
    )
    linf_budgets_255: list[float] = Field(default=[8], min_length=1, max_length=10)
    l2_budgets: list[float] = Field(default=[0.5], min_length=1, max_length=10)
    batch_size: int = Field(default=32, ge=1, le=512)
    sample_limit: int = Field(default=0, ge=0, le=1000000)
    seed: int = Field(default=0, ge=0, le=2000000000)
    step_size_255: float = Field(default=1, gt=0, le=32)
    steps: int = Field(default=20, ge=1, le=500)
    restarts: int = Field(default=1, ge=1, le=20)
    cw_steps: int = Field(default=100, ge=1, le=500)
    cw_confidence: float = Field(default=0, ge=0, le=100)
    deepfool_steps: int = Field(default=50, ge=1, le=500)
    deepfool_candidates: int = Field(default=10, ge=2, le=1000)
    deepfool_overshoot: float = Field(default=0.02, ge=0, le=1)

    @model_validator(mode="after")
    def valid_grid(self):
        if len(set(self.model_ids)) != len(self.model_ids) or len(
            set(self.attacks)
        ) != len(self.attacks):
            raise ValueError("模型或攻击列表不能重复。")
        if any(
            value < 0 or value > 32 or not math.isfinite(value)
            for value in self.linf_budgets_255
        ):
            raise ValueError("L∞ 预算必须在 0—32/255 内。")
        if any(
            value < 0 or value > 20 or not math.isfinite(value)
            for value in self.l2_budgets
        ):
            raise ValueError("L2 预算必须在 0—20 内。")
        if len(set(self.linf_budgets_255)) != len(self.linf_budgets_255) or len(
            set(self.l2_budgets)
        ) != len(self.l2_budgets):
            raise ValueError("预算不能重复。")
        return self


def save_json(path, body):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def digest_json(body):
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def attack_grid(config):
    common = {
        name: getattr(config, name)
        for name in AttackConfig.model_fields
        if name not in ("algorithm", "epsilon_255", "epsilon_l2")
    }
    result = []
    for name in config.attacks:
        norm = CATALOG[name]["norm"]
        for budget in config.l2_budgets if norm == "L2" else config.linf_budgets_255:
            options = dict(common, algorithm=name)
            options["epsilon_l2" if norm == "L2" else "epsilon_255"] = budget
            result.append(AttackConfig(**options))
    return result


def validate_plan(config, dataset_spec, model_specs):
    if dataset_spec.get("status") != "validated":
        raise ValueError("请先检查数据集。")
    for model_spec in model_specs:
        if model_spec.get("status") != "validated":
            raise ValueError("所有模型都必须先通过验证。")
        check_dataset_model(dataset_spec, model_spec)
        for attack_name in config.attacks:
            check_attack_compatibility(
                AttackConfig(algorithm=attack_name), model_spec["num_classes"]
            )


def summarize_rows(rows, number_of_classes):
    clean_count = sum(row["clean_correct"] for row in rows)
    robust_count = sum(row["robust_correct"] for row in rows)
    per_class = []
    for class_index in range(number_of_classes):
        selected = [row for row in rows if row["label"] == class_index]
        count = len(selected)
        per_class.append(
            {
                "class_index": class_index,
                "count": count,
                "clean_accuracy": (
                    sum(row["clean_correct"] for row in selected) / count
                    if count
                    else None
                ),
                "robust_accuracy": (
                    sum(row["robust_correct"] for row in selected) / count
                    if count
                    else None
                ),
            }
        )
    observed = [item for item in per_class if item["count"]]
    return {
        "sample_count": len(rows),
        "clean_correct_count": clean_count,
        "clean_accuracy": clean_count / len(rows),
        "robust_accuracy": robust_count / len(rows),
        "attack_success_rate_on_clean_correct": (
            (clean_count - robust_count) / clean_count if clean_count else None
        ),
        "macro_clean_accuracy": sum(item["clean_accuracy"] for item in observed)
        / len(observed),
        "macro_robust_accuracy": sum(item["robust_accuracy"] for item in observed)
        / len(observed),
        "per_class": per_class,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_suite(
    config, dataset_spec, model_specs, directory, progress=None, cancelled=None
):
    """缓存必须同时匹配配置、权重、数据、代码及实际批输入，才允许恢复。"""
    if not isinstance(config, BatchConfig):
        config = BatchConfig.model_validate(config)
    validate_plan(config, dataset_spec, model_specs)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    current_inspection = inspect_dataset(dataset_spec)
    if (
        current_inspection["source_fingerprint"]
        != dataset_spec["inspection"]["source_fingerprint"]
    ):
        raise ValueError("数据目录在登记后发生变化，请重新检查数据集。")
    snapshots = []
    for model_spec in model_specs:
        actual_hash = engine.sha256(model_spec["weights_path"])
        if actual_hash != model_spec["validation"]["sha256"]:
            raise ValueError("模型权重已变化，请重新验证模型。")
        snapshot = dict(model_spec, weights_sha256=actual_hash)
        if model_spec["architecture"] == "custom":
            snapshot["loader_sha256"] = engine.sha256(model_spec["loader_path"])
            if snapshot["loader_sha256"] != model_spec["validation"].get(
                "loader_sha256"
            ):
                raise ValueError("自定义模型加载文件已变化或尚未校验，请重新验证模型。")
        snapshots.append(snapshot)
    code_hashes = {
        name: engine.sha256(Path(__file__).parent / name)
        for name in (
            "attacks.py",
            "attack_protocol.py",
            "engine.py",
            "data_sources.py",
            "batch_engine.py",
        )
    }
    manifest = {
        "config": config.model_dump(),
        "dataset": dataset_spec,
        "models": snapshots,
        "dependencies": package_versions(),
        "code_sha256": code_hashes,
    }
    signature = digest_json(manifest)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError(
                "恢复条件已改变（配置、代码、权重、数据或版本），请新建实验。"
            )
    else:
        save_json(manifest_path, {"signature": signature, **manifest})
    total_samples = current_inspection["sample_count"]
    selected_indices = list(range(total_samples))
    if config.sample_limit and config.sample_limit < total_samples:
        generator = torch.Generator().manual_seed(config.seed)
        selected_indices = torch.randperm(total_samples, generator=generator)[
            : config.sample_limit
        ].tolist()
    save_json(directory / "sample_indices.json", selected_indices)
    grid = attack_grid(config)
    batches_per_cell = math.ceil(len(selected_indices) / config.batch_size)
    total_batches = len(grid) * len(model_specs) * batches_per_cell
    finished_batches = 0
    report = {
        "signature": signature,
        "dataset_name": dataset_spec["name"],
        "dataset_split": dataset_spec["split"],
        "classes": current_inspection["classes"],
        "sample_count": len(selected_indices),
        "subset": len(selected_indices) < total_samples,
        "cells": [],
        "combined": [],
        "manifest_file": "manifest.json",
        "note": "联合结果按同一模型、范数和预算逐样本取所有攻击的失败并集；不同范数不混合。",
    }
    all_csv_rows = []
    combined_rows = {}
    for model_spec in model_specs:
        if cancelled and cancelled():
            raise engine.Cancelled()
        model, device = engine.load_model(model_spec)
        data = load_dataset(dataset_spec, model_spec["input_size"])
        loader = DataLoader(
            Subset(data, selected_indices),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        for attack_config in grid:
            protocol = attack_config.protocol()
            cell_id = digest_json({"model": model_spec["id"], "protocol": protocol})[
                :16
            ]
            cell_directory = directory / cell_id
            cell_directory.mkdir(exist_ok=True)
            rows, elapsed, reused_batches, examples = [], 0.0, 0, []
            for batch_index, (images, labels) in enumerate(loader):
                if cancelled and cancelled():
                    raise engine.Cancelled()
                batch_path = cell_directory / f"batch_{batch_index:06d}.json"
                input_hash = hashlib.sha256(
                    images.numpy().tobytes() + labels.numpy().tobytes()
                ).hexdigest()
                if batch_path.exists():
                    saved = json.loads(batch_path.read_text(encoding="utf-8"))
                    if saved["input_sha256"] != input_hash:
                        raise ValueError("已完成批次的实际输入发生变化，不能复用。")
                    batch_rows = saved["rows"]
                    protocol = saved["protocol"]
                    elapsed += saved["seconds"]
                    examples.extend(saved.get("examples", []))
                    reused_batches += 1
                else:
                    images, labels = images.to(device), labels.to(device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.no_grad():
                        clean_predictions = engine.forward(
                            model, images, model_spec
                        ).argmax(1)
                    seeded = attack_config.model_copy(
                        update={"seed": config.seed + batch_index}
                    )
                    adversarial, details = run_attack(
                        model,
                        images,
                        labels,
                        model_spec,
                        seeded,
                        cancelled=cancelled,
                        log_path=(
                            cell_directory / f"aa_{batch_index:06d}.log"
                            if attack_config.algorithm == "autoattack"
                            else None
                        ),
                    )
                    with torch.no_grad():
                        adversarial_predictions = engine.forward(
                            model, adversarial, model_spec
                        ).argmax(1)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    batch_seconds = time.perf_counter() - started
                    elapsed += batch_seconds
                    batch_rows, batch_examples = [], []
                    for local_index in range(len(labels)):
                        sample_index = selected_indices[
                            batch_index * config.batch_size + local_index
                        ]
                        label = int(labels[local_index])
                        clean_prediction = int(clean_predictions[local_index])
                        adversarial_prediction = int(
                            adversarial_predictions[local_index]
                        )
                        delta = (
                            adversarial[local_index] - images[local_index]
                        ).flatten()
                        row = {
                            "sample_index": sample_index,
                            "label": label,
                            "clean_prediction": clean_prediction,
                            "adversarial_prediction": adversarial_prediction,
                            "clean_correct": clean_prediction == label,
                            "robust_correct": clean_prediction == label
                            and adversarial_prediction == label,
                            "linf": float(delta.abs().max()),
                            "l2": float(delta.norm()),
                        }
                        batch_rows.append(row)
                        if (
                            clean_prediction == label
                            and adversarial_prediction != label
                            and len(examples) + len(batch_examples) < 3
                        ):
                            prefix = f"case_{sample_index}"
                            engine.save_tensor(
                                images[local_index : local_index + 1],
                                cell_directory / f"{prefix}_original.png",
                            )
                            engine.save_tensor(
                                adversarial[local_index : local_index + 1],
                                cell_directory / f"{prefix}_adversarial.png",
                            )
                            batch_examples.append(
                                {
                                    "sample_index": sample_index,
                                    "label": label,
                                    "original": f"{cell_id}/{prefix}_original.png",
                                    "adversarial": f"{cell_id}/{prefix}_adversarial.png",
                                }
                            )
                    save_json(
                        batch_path,
                        {
                            "input_sha256": input_hash,
                            "rows": batch_rows,
                            "seconds": batch_seconds,
                            "examples": batch_examples,
                            "protocol": details["protocol"],
                            "seed": seeded.seed,
                        },
                    )
                    examples.extend(batch_examples)
                    protocol = details["protocol"]
                rows.extend(batch_rows)
                finished_batches += 1
                if progress:
                    progress(
                        round(finished_batches / total_batches * 99),
                        f"{model_spec['name']} / {protocol['name']} / {protocol['norm']} ε={protocol['budget']:g} / 批次 {batch_index+1}/{batches_per_cell}",
                        report,
                    )
            protocol["seed"] = attack_config.seed
            protocol["batch_seed_rule"] = "base_seed + batch_index"
            summary = summarize_rows(rows, model_spec["num_classes"])
            cell = {
                "id": cell_id,
                "model_id": model_spec["id"],
                "model_name": model_spec["name"],
                "protocol": protocol,
                "metrics": summary,
                "seconds": round(elapsed, 4),
                "reused_batches": reused_batches,
                "examples": examples,
            }
            report["cells"].append(cell)
            write_csv(cell_directory / "samples.csv", rows)
            for row in rows:
                all_csv_rows.append(
                    {
                        "model_id": model_spec["id"],
                        "model_name": model_spec["name"],
                        "attack": protocol["name"],
                        "norm": protocol["norm"],
                        "budget": protocol["budget"],
                        **row,
                    }
                )
            group_key = (model_spec["id"], protocol["norm"], protocol["budget"])
            if group_key not in combined_rows:
                combined_rows[group_key] = {
                    "model": model_spec,
                    "rows": {row["sample_index"]: dict(row) for row in rows},
                    "attacks": [],
                }
            else:
                saved_rows = combined_rows[group_key]["rows"]
                for row in rows:
                    baseline = saved_rows[row["sample_index"]]
                    if (
                        baseline["label"] != row["label"]
                        or baseline["clean_prediction"] != row["clean_prediction"]
                    ):
                        raise ValueError(
                            "不同攻击间的正常预测不一致，不能直接合并；请检查随机化与输入处理。"
                        )
                    saved_rows[row["sample_index"]]["robust_correct"] &= row[
                        "robust_correct"
                    ]
            combined_rows[group_key]["attacks"].append(protocol["name"])
            save_json(directory / "report.json", report)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for (model_id, norm, budget), group in combined_rows.items():
        report["combined"].append(
            {
                "model_id": model_id,
                "model_name": group["model"]["name"],
                "norm": norm,
                "budget": budget,
                "attacks": group["attacks"],
                "metrics": summarize_rows(
                    list(group["rows"].values()), group["model"]["num_classes"]
                ),
            }
        )
    write_csv(directory / "samples.csv", all_csv_rows)
    summary_rows = [
        {
            "model": cell["model_name"],
            "attack": cell["protocol"]["name"],
            "norm": cell["protocol"]["norm"],
            "budget": cell["protocol"]["budget"],
            "sample_count": cell["metrics"]["sample_count"],
            "clean_accuracy": cell["metrics"]["clean_accuracy"],
            "robust_accuracy": cell["metrics"]["robust_accuracy"],
            "attack_success_rate": cell["metrics"][
                "attack_success_rate_on_clean_correct"
            ],
            "seconds": cell["seconds"],
        }
        for cell in report["cells"]
    ]
    write_csv(directory / "summary.csv", summary_rows)
    class_rows = [
        {
            **{k: summary_rows[i][k] for k in ("model", "attack", "norm", "budget")},
            **item,
            "class_name": current_inspection["classes"][item["class_index"]],
        }
        for i, cell in enumerate(report["cells"])
        for item in cell["metrics"]["per_class"]
    ]
    write_csv(directory / "per_class.csv", class_rows)
    save_json(directory / "report.json", report)
    from .reporting import write_html_report

    write_html_report(report, directory / "report.html")
    return report
