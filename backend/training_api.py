"""训练任务通过独立 Python 进程执行；网页只提交配置、读取真实日志。"""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from . import store, engine
from .training_config import TrainingConfig
from .model_schema import model_payload
from .jobs import JobQueue, require, now


class TrainingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    config: dict
    teacher_model_id: str = ""


def read_metrics(path):
    rows = []
    if Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # 正在写入的最后一行留待下一次轮询读取。
                continue
    return rows


def training_worker(body, resuming=False):
    def worker(key, cancelled, progress):
        folder = store.DATA / key
        folder.mkdir(exist_ok=True)
        config = body["config"]
        if body.get("teacher_model_id"):
            teacher = require("models", body["teacher_model_id"])
            if teacher["status"] != "validated":
                raise ValueError("教师模型需要先通过验证。")
            (folder / "teacher_model.json").write_text(
                json.dumps(model_payload(teacher), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        path = folder / "training_config.json"
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        stop_file = folder / "stop.request"
        if stop_file.exists():
            stop_file.unlink()
        arguments = [
            sys.executable,
            "-m",
            "backend.train",
            "--config",
            str(path),
            "--stop-file",
            str(stop_file),
        ]
        checkpoint = folder / "training" / "last.pt"
        if resuming:
            if not checkpoint.is_file():
                # 如果在第一个完整 epoch 前中断，可以干净重启该任务的训练目录。
                raise ValueError("尚无完整 epoch 检查点；请复制配置创建新任务。")
            arguments.extend(["--resume", str(checkpoint)])
        environment = dict(
            os.environ,
            AEGIS_DEVICE="cuda",
            PYTHONUNBUFFERED="1",
            PYTHONIOENCODING="utf-8",
        )
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        log_path = folder / "training.log"
        with open(log_path, "a", encoding="utf-8") as log:
            process = subprocess.Popen(
                arguments,
                cwd=store.ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
            store.update("training_runs", key, pid=process.pid)
            try:
                while process.poll() is None:
                    if cancelled():
                        stop_file.touch(exist_ok=True)
                    metrics = read_metrics(folder / "training" / "metrics.jsonl")
                    # 完成 checkpoint 后才计为完成 epoch；显示已有真实日志行。
                    epoch = max((item["epoch"] for item in metrics), default=0)
                    progress(
                        min(99, round(epoch / config["epochs"] * 100)),
                        f'已记录 {epoch}/{config["epochs"]} 个 epoch；停止请求会在批次边界检查。',
                    )
                    time.sleep(0.5)
                code = process.wait()
            finally:
                if process.poll() is None:
                    stop_file.touch(exist_ok=True)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=15)
        if cancelled() or code == 130:
            raise engine.Cancelled()
        if code != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"训练进程退出码 {code}。\n{tail}")
        return dict(
            metrics=read_metrics(folder / "training" / "metrics.jsonl"),
            registration_file="training/model_registration.json",
            checkpoint_file="training/last.pt",
            weights_file="training/best_weights.pt",
        )

    return worker


def install_training(app, pool, cancellations):
    queue = JobQueue(pool, cancellations)

    @app.get("/api/training-runs")
    def list_training():
        return [
            {key: value for key, value in item.items() if key != "report"}
            for item in store.all_items("training_runs")
        ]

    @app.post("/api/training-runs", status_code=201)
    def create_training(body: TrainingInput):
        key = str(uuid.uuid4())
        config = dict(body.config, output_dir=str(store.DATA / key / "training"))
        if body.teacher_model_id:
            require("models", body.teacher_model_id)
            config["teacher_model_file"] = str(store.DATA / key / "teacher_model.json")
        try:
            config = TrainingConfig.model_validate(config).model_dump()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return store.save(
            "training_runs",
            dict(
                id=key,
                name=body.name,
                config=config,
                teacher_model_id=body.teacher_model_id,
                status="configured",
                progress=0,
                created_at=now(),
                error=None,
                message="配置已保存，尚未开始训练。",
            ),
        )

    def launch(key, resuming):
        body = require("training_runs", key)
        if key in cancellations or body["status"] in (
            "queued",
            "running",
            "cancelling",
            "succeeded",
        ):
            raise HTTPException(409, "当前状态不能开始或继续训练。")
        try:
            if engine.select_device().type != "cuda":
                raise ValueError("训练任务始终要求 CUDA；CPU 仅用于独立损失/接口测试。")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if resuming and not (store.DATA / key / "training" / "last.pt").is_file():
            raise HTTPException(409, "尚无完整 epoch 检查点，请复制配置创建新任务。")
        if not resuming and body["status"] != "configured":
            raise HTTPException(
                409, "失败/中断任务请用继续训练，或复制配置创建新任务。"
            )
        return queue.submit(
            "training_runs", body, training_worker(body, resuming), resume=True
        )

    @app.post("/api/training-runs/{key}/start", status_code=202)
    def start(key: str):
        return launch(key, False)

    @app.post("/api/training-runs/{key}/resume", status_code=202)
    def resume(key: str):
        return launch(key, True)

    @app.get("/api/training-runs/{key}/telemetry")
    def telemetry(key: str):
        job = require("training_runs", key)
        folder = store.DATA / key
        log_path = folder / "training.log"
        return dict(
            job=job,
            metrics=read_metrics(folder / "training" / "metrics.jsonl"),
            log=(
                log_path.read_text(encoding="utf-8", errors="replace")[-16000:]
                if log_path.is_file()
                else ""
            ),
            checkpoint_available=(folder / "training" / "last.pt").is_file(),
        )

    @app.post("/api/training-runs/{key}/import-model", status_code=201)
    def import_model(key: str):
        job = require("training_runs", key)
        path = store.DATA / key / "training" / "model_registration.json"
        if job["status"] != "succeeded" or not path.is_file():
            raise HTTPException(409, "成功训练并生成模型后才可以导入。")
        if job.get("imported_model_id"):
            return require("models", job["imported_model_id"])
        payload = model_payload(json.loads(path.read_text(encoding="utf-8")))
        model = store.save(
            "models",
            dict(
                payload,
                id=str(uuid.uuid4()),
                status="registered",
                validation=None,
                created_at=now(),
                training_run_id=key,
            ),
        )
        store.update("training_runs", key, imported_model_id=model["id"])
        return model
