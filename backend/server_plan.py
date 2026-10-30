"""迁移后顺序执行训练、验证集导出、攻击评测和防御评测；本机不自动执行。"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from . import engine, store
from .training_config import load_configuration
from .training_data import DATASETS
from .batch_engine import BatchConfig, run_suite, save_json
from .data_sources import inspect_dataset
from .defenses import DefenseConfig
from .defense_evaluation import (
    DefenseEvaluationConfig,
    run_defense_suite,
    calibrate,
    calibration_signature,
)
from .export_validation import export_validation
from .server_check import check_environment
from .jobs import now
from .plan_results import run_logged_training, publish_training, publish_evaluation


def main():
    parser = argparse.ArgumentParser(description="Aegis Lab GPU 服务器执行计划")
    parser.add_argument("--plan", default="configs/server-plan.json")
    parser.add_argument(
        "--execute", action="store_true", help="显式执行，否则只打印计划"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="独立目录，1 epoch / 少量样本检查，结果不能用于性能结论",
    )
    parser.add_argument(
        "--sample-limit", type=int, default=100, help="评测随机子集；0 代表完整测试划分"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="评测批大小")
    parser.add_argument(
        "--download", action="store_true", help="显式允许下载 CIFAR 数据"
    )
    parser.add_argument(
        "--run-label", default="main", help="独立实验组标识；改变训练配置时使用新标识"
    )
    args = parser.parse_args()
    if not re.fullmatch("[a-z0-9_-]{1,24}", args.run_label):
        raise ValueError("run-label 需要 1-24 个小写字母、数字、下划线或短横线。")
    if args.sample_limit < 0 or not 1 <= args.batch_size <= 512:
        raise ValueError("样本数或批大小无效。")
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    base = plan_path.parent
    runs = []
    for item in plan["training_runs"]:
        if not re.fullmatch("[a-z0-9_-]+", item["id"]):
            raise ValueError("训练 ID 只允许小写字母、数字、下划线或短横线。")
        runs.append((item, load_configuration(base / item["config"])))
    if len({item[0]["id"] for item in runs}) != len(runs):
        raise ValueError("训练 ID 重复。")
    if len({config["dataset"] for _, config in runs}) != 1 or runs[0][1][
        "dataset"
    ] not in ("cifar10", "cifar100"):
        raise ValueError(
            "此示例批量计划要求同一个 CIFAR 数据集；ImageFolder 使用独立训练/评测入口。"
        )
    seen = set()
    for item, config in runs:
        if config["method"] == "distill" and item.get("teacher_run_id") not in seen:
            raise ValueError("学生的 teacher_run_id 必须位于计划中更早的训练项。")
        seen.add(item["id"])
    for definition in plan["defenses"]:
        options = DefenseConfig.model_validate(definition)
        if not set(options.model_ids) <= seen:
            raise ValueError("防御配置引用了不在训练计划中的模型 ID。")
    root = (
        store.DATA / ("server-smoke" if args.smoke else "server-run") / args.run_label
    ).resolve()
    print(
        json.dumps(
            dict(
                output_root=str(root),
                training=[
                    dict(id=item["id"], method=config["method"], config=item["config"])
                    for item, config in runs
                ],
                evaluation_sample_limit=8 if args.smoke else args.sample_limit,
                execute=args.execute,
                smoke=args.smoke,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not args.execute:
        return
    if engine.select_device().type != "cuda":
        raise ValueError("服务器计划执行要求 CUDA。")
    check = check_environment(gpu_smoke=True)
    if not check["ok"]:
        raise ValueError("服务器依赖或 GPU 自检未通过。")
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "server_check.json", check)
    store.initialize()
    models = []
    outputs = {}
    aliases = {}
    mode = ("smoke" if args.smoke else "full") + "-" + args.run_label
    training_job_ids = []
    for item, config in runs:
        name = item["id"]
        job_id = f"server-{mode}-training-{name}"
        training_job_ids.append(job_id)
        output = store.DATA / job_id / "training"
        config["output_dir"] = str(output)
        config["download"] = args.download
        if args.smoke:
            number_of_classes = DATASETS[config["dataset"]]["classes"]
            config.update(
                epochs=1,
                batch_size=min(config["batch_size"], 16),
                attack_steps=2,
                validation_steps=2,
                num_workers=0,
                max_train_samples=max(64, number_of_classes),
                max_validation_samples=max(32, number_of_classes),
            )
        if config["method"] == "distill":
            teacher_id = item.get("teacher_run_id")
            if teacher_id not in outputs:
                raise ValueError("学生之前必须完成其 teacher_run_id 对应训练。")
            config["teacher_model_file"] = str(
                outputs[teacher_id] / "model_registration.json"
            )
        config_path = root / f"training-{name}.json"
        save_json(config_path, config)
        arguments = [
            sys.executable,
            "-m",
            "backend.train",
            "--config",
            str(config_path),
        ]
        if (output / "last.pt").is_file():
            arguments.extend(["--resume", str(output / "last.pt")])
        print(f"开始 GPU 训练 / 恢复：{name}", flush=True)
        started_at = now()
        run_logged_training(arguments, job_id)
        outputs[name] = output
        spec = json.loads(
            (output / "model_registration.json").read_text(encoding="utf-8")
        )
        spec.update(
            id=f"server-{mode}-{name}", status="validated", training_run_id=job_id
        )
        aliases[name] = spec["id"]
        if args.smoke:
            spec["name"] += " / SMOKE 功能检查"
        spec["validation"] = engine.validate_model(spec)
        spec["validation"].pop("validation_seconds", None)
        models.append(spec)
        store.save("models", spec)
        publish_training(job_id, spec["name"], config, spec["id"], started_at)
    first_config = root / f'training-{runs[0][0]["id"]}.json'
    validation = export_validation(
        first_config,
        outputs[runs[0][0]["id"]] / "split.json",
        root / "calibration-images",
    )
    validation.update(status="validated", id=f"server-{mode}-validation")
    validation["inspection"] = inspect_dataset(validation)
    store.save("datasets", validation)
    config = runs[0][1]
    dataset = dict(
        id=f"server-{mode}-official-test",
        name=config["dataset"]
        + " 官方测试集"
        + (" / SMOKE 功能检查" if args.smoke else ""),
        kind=config["dataset"],
        root=str(Path(config["data_dir"]).resolve()),
        split="test",
        status="validated",
    )
    dataset["inspection"] = inspect_dataset(dataset)
    store.save("datasets", dataset)
    limit = 8 if args.smoke else args.sample_limit
    suite = BatchConfig(
        dataset_id=dataset["id"],
        model_ids=[item["id"] for item in models],
        attacks=["pgd20", "pgd100", "cw_linf", "autoattack", "deepfool_linf"],
        sample_limit=limit,
        batch_size=args.batch_size,
    )
    save_json(
        root / "evaluation-suite.json",
        dict(experiment=suite.model_dump(), dataset=dataset, models=models),
    )

    def progress(value, message, report=None):
        print(f"{value}% {message}", flush=True)

    attack_job = f"server-{mode}-attacks-n{limit}-b{args.batch_size}"
    started_at = now()
    attack_report = run_suite(suite, dataset, models, store.DATA / attack_job, progress)
    publish_evaluation(
        "batch_runs", attack_job, suite, dataset, attack_report, models, started_at
    )
    profiles = []
    members = {}
    for index, definition in enumerate(plan["defenses"]):
        ids = [aliases[name] for name in definition["model_ids"]]
        specs = [next(item for item in models if item["id"] == key) for key in ids]
        profile_config = DefenseConfig.model_validate(
            dict(definition, model_ids=ids)
        ).model_dump()
        if args.smoke:
            profile_config["name"] += " / SMOKE 功能检查"
        key = f"server-{mode}-defense-{index}"
        profile = dict(
            id=key,
            name=profile_config["name"],
            model_ids=ids,
            config=profile_config,
            calibration=None,
        )
        if profile_config["detector"]["method"] != "none":
            result = calibrate(
                profile,
                specs,
                validation,
                root / "calibration" / key,
                sample_limit=32 if args.smoke else 0,
                batch_size=args.batch_size,
                progress=progress,
            )
            profile["config"]["detector"]["threshold"] = max(
                0, min(1, result["threshold"])
            )
            profile["calibration"] = dict(
                result,
                applied_signature=calibration_signature(profile["config"], specs),
            )
        profiles.append(profile)
        members[key] = specs
        store.save("defenses", profile)
    defense_options = DefenseEvaluationConfig(
        dataset_id=dataset["id"],
        defense_ids=[item["id"] for item in profiles],
        attacks=["pgd20", "pgd100", "cw_linf"],
        sample_limit=limit,
        batch_size=args.batch_size,
    )
    save_json(
        root / "defense-suite.json",
        dict(
            experiment=defense_options.model_dump(),
            dataset=dataset,
            models=models,
            defenses=profiles,
        ),
    )
    defense_job = f"server-{mode}-defense-evaluation-n{limit}-b{args.batch_size}"
    started_at = now()
    defense_report = run_defense_suite(
        defense_options, dataset, profiles, members, store.DATA / defense_job, progress
    )
    publish_evaluation(
        "defense_runs",
        defense_job,
        defense_options,
        dataset,
        defense_report,
        models,
        started_at,
    )
    save_json(
        root / "result-index.json",
        dict(
            training_run_ids=training_job_ids,
            attack_run_id=attack_job,
            defense_run_id=defense_job,
            models=[item["id"] for item in models],
            profiles=[item["id"] for item in profiles],
            note="结果与真实文件已登记，可启动网页查看；SMOKE 与正式计划分别保存。",
        ),
    )
    print(
        "服务器计划完成。结果在 " + str(root) + "；本次是否子集请查看报告。", flush=True
    )


if __name__ == "__main__":
    main()
