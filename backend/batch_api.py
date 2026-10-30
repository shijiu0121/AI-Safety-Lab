"""数据集和批量实验接口。与单图实验共用一个计算队列，避免显存争抢。"""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

from . import engine, store
from .attack_protocol import CATALOG
from .attacks import package_versions
from .batch_engine import BatchConfig, run_suite, validate_plan
from .data_sources import DatasetInput, inspect_dataset


def now():
    return datetime.now(timezone.utc).isoformat()


def install_routes(app, pool, cancellations):
    def require_record(table, key):
        record = store.get(table, key)
        if record is None:
            raise HTTPException(404, "记录不存在。")
        return record

    def run_job(key):
        event = cancellations[key]
        try:
            if event.is_set():
                raise engine.Cancelled()
            job = require_record("batch_runs", key)
            config = BatchConfig.model_validate(job["config"])
            dataset_spec = require_record("datasets", config.dataset_id)
            model_specs = [
                require_record("models", model_id) for model_id in config.model_ids
            ]
            store.update(
                "batch_runs", key, status="running", started_at=now(), error=None
            )

            def update_progress(value, message, partial_report):
                store.update(
                    "batch_runs",
                    key,
                    progress=value,
                    message=message,
                    report=partial_report,
                )

            report = run_suite(
                config,
                dataset_spec,
                model_specs,
                store.DATA / key,
                update_progress,
                event.is_set,
            )
            if event.is_set():
                raise engine.Cancelled()
            store.update(
                "batch_runs",
                key,
                status="succeeded",
                progress=100,
                report=report,
                finished_at=now(),
                message="全部组合已完成。",
            )
        except engine.Cancelled:
            store.update(
                "batch_runs",
                key,
                status="cancelled",
                finished_at=now(),
                message="已取消；已完成批次可在条件未改变时恢复。",
            )
        except Exception as exc:
            store.update(
                "batch_runs", key, status="failed", error=str(exc), finished_at=now()
            )
        finally:
            cancellations.pop(key, None)

    @app.get("/api/attacks")
    def list_attacks():
        versions = package_versions()
        return [
            {
                "id": key,
                **value,
                "available": value["backend"] == "PyTorch"
                or bool(
                    versions[
                        "foolbox" if value["backend"] == "Foolbox" else "autoattack"
                    ]
                ),
                "min_classes": 10 if key == "autoattack" else 2,
            }
            for key, value in CATALOG.items()
        ]

    @app.get("/api/datasets")
    def list_datasets():
        return [
            item for item in store.all_items("datasets") if not item.get("archived")
        ]

    @app.post("/api/datasets", status_code=201)
    def create_dataset(config: DatasetInput):
        return store.save(
            "datasets",
            dict(
                config.model_dump(),
                id=str(uuid.uuid4()),
                status="registered",
                inspection=None,
                created_at=now(),
            ),
        )

    @app.post("/api/datasets/{key}/inspect")
    def inspect(key: str):
        spec = require_record("datasets", key)
        try:
            inspection = inspect_dataset(spec)
            return store.update(
                "datasets", key, status="validated", inspection=inspection, error=None
            )
        except Exception as exc:
            store.update("datasets", key, status="invalid", error=str(exc))
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/batch-runs")
    def list_batch_runs():
        # 列表不返回整份报告；详情页按需读取，避免轮询时重复传输大量逐类数据。
        return [
            {k: v for k, v in job.items() if k != "report"}
            for job in store.all_items("batch_runs")
        ]

    @app.get("/api/batch-runs/{key}")
    def get_job(key: str):
        return require_record("batch_runs", key)

    @app.post("/api/batch-runs", status_code=202)
    def create_job(config: BatchConfig):
        if len(cancellations) >= 8:
            raise HTTPException(429, "队列已满，请稍后提交。")
        dataset_spec = require_record("datasets", config.dataset_id)
        model_specs = [require_record("models", key) for key in config.model_ids]
        try:
            validate_plan(config, dataset_spec, model_specs)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        key = str(uuid.uuid4())
        job = {
            "id": key,
            "config": config.model_dump(),
            "dataset_name": dataset_spec["name"],
            "model_names": [model["name"] for model in model_specs],
            "created_at": now(),
            "status": "queued",
            "progress": 0,
            "report": None,
            "error": None,
            "message": "等待计算资源。",
        }
        store.save("batch_runs", job)
        cancellations[key] = threading.Event()
        pool.submit(run_job, key)
        return job

    @app.post("/api/batch-runs/{key}/cancel")
    def cancel_job(key: str):
        job = require_record("batch_runs", key)
        event = cancellations.get(key)
        if event:
            event.set()
            return store.update("batch_runs", key, status="cancelling")
        return job

    @app.post("/api/batch-runs/{key}/resume", status_code=202)
    def resume_job(key: str):
        job = require_record("batch_runs", key)
        if (
            job["status"] not in ("failed", "cancelled", "interrupted")
            or key in cancellations
        ):
            raise HTTPException(409, "只有失败、取消或中断的任务可以恢复。")
        if len(cancellations) >= 8:
            raise HTTPException(429, "队列已满。")
        cancellations[key] = threading.Event()
        store.update(
            "batch_runs", key, status="queued", error=None, message="准备检查恢复条件。"
        )
        pool.submit(run_job, key)
        return require_record("batch_runs", key)

    @app.get("/api/batch-runs/{key}/files/{filename:path}")
    def output_file(key: str, filename: str):
        require_record("batch_runs", key)
        directory = (store.DATA / key).resolve()
        target = (directory / filename).resolve()
        if (
            not target.is_relative_to(directory)
            or target.suffix not in (".json", ".csv", ".png", ".log", ".html")
            or not target.is_file()
        ):
            raise HTTPException(404, "结果文件不存在或不允许访问。")
        # 结果文件仅限本任务目录；不接受目录跳转或任意本机文件。
        return FileResponse(
            target, filename=target.name if target.suffix != ".png" else None
        )
