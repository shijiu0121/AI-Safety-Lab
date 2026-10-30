"""命令行入口。"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="aegis")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="训练单个模型")
    train.add_argument("--config", required=True)
    train.add_argument("--resume")
    plan = commands.add_parser("experiment", help="展开或运行统一实验计划")
    default_plan = Path("configs/experiment-plan.json")
    if not default_plan.exists():
        default_plan = Path(__file__).parent / "configs" / "experiment-plan.json"
    plan.add_argument("--plan", default=str(default_plan))
    plan.add_argument("--stage", choices=["plan", "train", "evaluate", "detect", "defend", "anomaly", "all"], default="plan")
    plan.add_argument("--group", choices=["all", "main", "long_tail"], default="all")
    plan.add_argument("--only", default="", help="按任务 ID 的子串筛选")
    plan.add_argument("--download", action="store_true")
    plan.add_argument("--smoke", action="store_true")
    export = commands.add_parser("export", help="导出成果网站数据")
    export.add_argument("--source", default="data/experiments-v04")
    export.add_argument("--destination", default="public/data")
    export.add_argument("--include-weights", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        from ._core.train import main as train_main
        sys.argv = ["aegis train", "--config", args.config] + (["--resume", args.resume] if args.resume else [])
        train_main()
    elif args.command == "experiment":
        from .experiments import execute_plan
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        result = execute_plan(plan, stage=args.stage, group=args.group, only=args.only, download=args.download, smoke=args.smoke)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        from .publishing import export_site
        print(json.dumps(export_site(args.source, args.destination, args.include_weights), ensure_ascii=False))


if __name__ == "__main__":
    main()
