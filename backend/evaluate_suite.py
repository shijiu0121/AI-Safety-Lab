"""从 JSON 配置启动与网页共用的批量实验；无需运行 Web 服务。"""

import argparse
import json
from pathlib import Path

from . import engine
from .batch_engine import BatchConfig, run_suite
from .data_sources import inspect_dataset


def main():
    parser = argparse.ArgumentParser(description="多模型、多攻击、多预算的 GPU 评测")
    parser.add_argument("--config", required=True, help="完整实验配置 JSON")
    parser.add_argument("--output-dir", required=True, help="相同配置可复用已完成批次")
    parser.add_argument(
        "--check-only", action="store_true", help="检查配置、模型、数据，不执行攻击"
    )
    args = parser.parse_args()
    specification = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = BatchConfig.model_validate(specification["experiment"])
    dataset_spec = specification["dataset"]
    dataset_spec.setdefault("id", "cli_dataset")
    dataset_spec["inspection"] = inspect_dataset(dataset_spec)
    dataset_spec["status"] = "validated"
    model_specs = specification["models"]
    for index, model_spec in enumerate(model_specs):
        model_spec.setdefault("id", f"model_{index}")
        model_spec.setdefault("categories", [])
        model_spec.setdefault("loader_path", "")
        model_spec.setdefault("factory", "build_model")
        model_spec["validation"] = engine.validate_model(model_spec)
        model_spec["status"] = "validated"
        # 验证时间不是实验协议，避免同配置命令行恢复被运行时长干扰。
        model_spec["validation"].pop("validation_seconds", None)
    from .batch_engine import validate_plan

    validate_plan(config, dataset_spec, model_specs)
    if (
        config.model_ids != [model["id"] for model in model_specs]
        or config.dataset_id != dataset_spec["id"]
    ):
        raise ValueError("experiment 中的 model_ids/dataset_id 必须与资源定义一致。")
    if args.check_only:
        print("GPU、模型、数据与实验配置检查通过。尚未执行攻击。")
        return

    def progress(value, message, report):
        print(f"{value}% {message}", flush=True)

    report = run_suite(
        config, dataset_spec, model_specs, Path(args.output_dir), progress
    )
    print(
        json.dumps(
            {"cells": len(report["cells"]), "sample_count": report["sample_count"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
