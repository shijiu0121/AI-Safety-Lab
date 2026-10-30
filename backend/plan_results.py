"""服务器命令行产物登记到网页；文件目录沿用网页任务约定。"""

import json
import os
import subprocess
import sys
from pathlib import Path
from . import store
from .jobs import now


def run_logged_training(arguments, job_id):
    directory = store.DATA / job_id
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "training.log", "a", encoding="utf-8") as log:
        with subprocess.Popen(
            arguments,
            cwd=store.ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"),
        ) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, arguments)


def publish_training(job_id, name, config, model_id, started_at):
    directory = store.DATA / job_id / "training"
    history = [
        json.loads(line)
        for line in (directory / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return store.save(
        "training_runs",
        dict(
            id=job_id,
            name=name,
            config=config,
            teacher_model_id="",
            status="succeeded",
            progress=100,
            created_at=started_at,
            started_at=started_at,
            finished_at=now(),
            message="服务器计划执行完成。",
            error=None,
            imported_model_id=model_id,
            report=dict(
                metrics=history,
                registration_file="training/model_registration.json",
                checkpoint_file="training/last.pt",
                weights_file="training/best_weights.pt",
            ),
        ),
    )


def publish_evaluation(kind, job_id, config, dataset, report, models, started_at):
    directory = store.DATA / job_id
    if not (directory / "report.json").is_file():
        raise ValueError("报告文件尚未生成，不能登记为完成。")
    return store.save(
        kind,
        dict(
            id=job_id,
            name="服务器计划 · "
            + ("攻防对比" if kind == "defense_runs" else "攻击评测"),
            config=config.model_dump(),
            dataset_name=dataset["name"],
            model_names=[item["name"] for item in models],
            status="succeeded",
            progress=100,
            created_at=started_at,
            started_at=started_at,
            finished_at=now(),
            message="服务器计划执行完成。",
            error=None,
            report=report,
        ),
    )
