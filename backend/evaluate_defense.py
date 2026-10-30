"""独立 GPU 防御评测 / 校准命令，与网页使用相同运行时。"""

import argparse
import json
from pathlib import Path
from . import engine
from .model_schema import model_payload
from .data_sources import DatasetInput, inspect_dataset
from .defenses import DefenseConfig, validate_members
from .defense_evaluation import (
    DefenseEvaluationConfig,
    run_defense_suite,
    calibrate,
    calibration_signature,
)


def prepare_specification(specification):
    models = {}
    for raw in specification["models"]:
        model = dict(model_payload(raw), id=raw["id"], status="validated")
        model["validation"] = engine.validate_model(model)
        model["validation"].pop("validation_seconds", None)
        models[model["id"]] = model
    raw_dataset = specification["dataset"]
    dataset = dict(
        DatasetInput.model_validate(
            {
                key: raw_dataset[key]
                for key in DatasetInput.model_fields
                if key in raw_dataset
            }
        ).model_dump(),
        id=raw_dataset["id"],
        status="validated",
    )
    dataset["inspection"] = inspect_dataset(dataset)
    profiles = []
    members = {}
    for raw in specification["defenses"]:
        config = DefenseConfig.model_validate(raw["config"])
        profile = dict(
            id=raw["id"],
            name=config.name,
            model_ids=config.model_ids,
            config=config.model_dump(),
            calibration=raw.get("calibration"),
        )
        selected = [models[key] for key in config.model_ids]
        validate_members(config, selected)
        profiles.append(profile)
        members[profile["id"]] = selected
    return dataset, profiles, members


def main():
    parser = argparse.ArgumentParser(description="Aegis Lab 防御评测与阈值校准")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--calibrate", default="", help="防御 ID；配置的数据集必须为独立 validation"
    )
    parser.add_argument("--target-fpr", type=float, default=0.05)
    args = parser.parse_args()
    specification = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = DefenseEvaluationConfig.model_validate(specification["experiment"])
    dataset, profiles, members = prepare_specification(specification)
    if args.check_only:
        print("GPU 模型与资源契约检查通过；尚未训练或攻击。")
        return

    def progress(value, message, report=None):
        print(f"{value}% {message}", flush=True)

    if args.calibrate:
        if not 0 < args.target_fpr < 0.5:
            raise ValueError("target-fpr 必须在 0 和 .5 之间。")
        profile = next(item for item in profiles if item["id"] == args.calibrate)
        result = calibrate(
            profile,
            members[profile["id"]],
            dataset,
            args.output_dir,
            args.target_fpr,
            config.sample_limit,
            config.batch_size,
            progress=progress,
        )
        profile["config"]["detector"]["threshold"] = max(0, min(1, result["threshold"]))
        profile["calibration"] = dict(
            result,
            applied_signature=calibration_signature(
                profile["config"], members[profile["id"]]
            ),
        )
        Path(args.output_dir, "calibrated_profile.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("校准完成；calibrated_profile.json 可替换评测配置的相应 defense 定义。")
    else:
        report = run_defense_suite(
            config, dataset, profiles, members, args.output_dir, progress
        )
        print(
            json.dumps(
                dict(cells=len(report["cells"]), sample_count=report["sample_count"]),
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
