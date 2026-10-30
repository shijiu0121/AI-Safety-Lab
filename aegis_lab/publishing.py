"""把实验目录转成网站可直接读取的 JSON、图片、报告和模型下载目录。"""
import csv
import html
import json
import shutil
from pathlib import Path
from .experiments import write_json


def export_site(source, destination, include_weights=False):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(source.glob("*/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        folder, run_id = path.parent, run["id"]
        output = destination / "downloads" / run_id
        output.mkdir(parents=True, exist_ok=True)
        record = dict(id=run_id, group=run["group"], dataset=run["config"]["dataset"], architecture=run["config"]["architecture"], method=run["config"]["method"], config=run["config"], command=run["command"], dependencies=run["dependencies"], downloads={})
        for name in ("config.json", "requested_config.json", "split.json", "evaluation_split.json", "model_registration.json", "metrics.jsonl", "evaluation.json", "detection.json", "defenses.json", "anomaly.json"):
            original = folder / name
            if original.exists():
                shutil.copy2(original, output / name)
                record["downloads"][name] = f"downloads/{run_id}/{name}"
        for directory in ("detectors", "anomaly"):
            if (folder / directory).exists():
                shutil.copytree(folder / directory, output / directory, dirs_exist_ok=True)
                for artifact in (folder / directory).glob("*.pt"):
                    record["downloads"][f"{directory}/{artifact.name}"] = f"downloads/{run_id}/{directory}/{artifact.name}"
        if include_weights and (folder / "best_weights.pt").exists():
            shutil.copy2(folder / "best_weights.pt", output / "best_weights.pt")
            record["downloads"]["weights"] = f"downloads/{run_id}/best_weights.pt"
        history = folder / "metrics.jsonl"
        record["history"] = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()] if history.exists() else []
        for name in ("evaluation", "detection", "defenses", "anomaly"):
            file = folder / f"{name}.json"
            record[name] = json.loads(file.read_text(encoding="utf-8")) if file.exists() else None
        if record["evaluation"]:
            for case in record["evaluation"]["cases"]:
                for key in ("original", "adversarial", "difference"):
                    # 通过文件名重新定位，实验目录从服务器复制回来后仍可导出。
                    image_path = folder / "cases" / Path(case[key].replace("\\", "/")).name
                    target = output / "cases" / image_path.name
                    target.parent.mkdir(exist_ok=True)
                    shutil.copy2(image_path, target)
                    case[key] = f"downloads/{run_id}/cases/{target.name}"
            results = record["evaluation"]["results"]
            csv_file = output / "metrics.csv"
            with csv_file.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=["attack", "epsilon_255", "clean_accuracy", "robust_accuracy", "balanced_robust_accuracy", "attack_success_rate", "samples", "milliseconds_per_sample"])
                writer.writeheader()
                for row in results:
                    writer.writerow({key: row["parameters"]["epsilon_255"] if key == "epsilon_255" else row[key] for key in writer.fieldnames})
            record["downloads"]["metrics.csv"] = f"downloads/{run_id}/metrics.csv"
        report = '<!doctype html><meta charset="utf-8"><title>' + html.escape(run_id) + '</title><style>body{max-width:1000px;margin:50px auto;font:16px system-ui}pre{white-space:pre-wrap;background:#f3f6f5;padding:20px}h1{color:#17664c}</style><h1>' + html.escape(run_id) + '</h1><pre>' + html.escape(json.dumps(record, ensure_ascii=False, indent=2)) + '</pre>'
        (output / "report.html").write_text(report, encoding="utf-8")
        record["downloads"]["report"] = f"downloads/{run_id}/report.html"
        records.append(record)
    manifest = dict(schema_version=1, experiments=records)
    write_json(destination / "results.json", manifest)
    if (source / "plan.json").exists():
        shutil.copy2(source / "plan.json", destination / "experiment-plan.json")
    return dict(experiments=len(records), destination=str(destination.resolve()))
