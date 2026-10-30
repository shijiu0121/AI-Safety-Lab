"""新增工作流复用已有单计算队列；任务状态写入 SQLite。"""

import threading
import uuid
from datetime import datetime, timezone
from fastapi import HTTPException
from . import engine, store


def now():
    return datetime.now(timezone.utc).isoformat()


def require(table, key):
    item = store.get(table, key)
    if item is None or item.get("archived"):
        raise HTTPException(404, "记录不存在或已归档。")
    return item


class JobQueue:
    def __init__(self, pool, cancellations):
        self.pool = pool
        self.cancellations = cancellations

    def submit(self, table, body, worker, resume=False):
        if len(self.cancellations) >= 8:
            raise HTTPException(429, "计算队列已满。")
        key = body.get("id") if resume else str(uuid.uuid4())
        event = threading.Event()
        body = dict(
            body,
            id=key,
            status="queued",
            progress=0,
            error=None,
            message="等待计算队列。",
        )
        if not resume:
            body["created_at"] = now()
        store.save(table, body)
        self.cancellations[key] = event

        def run():
            try:
                if event.is_set():
                    raise engine.Cancelled()
                store.update(table, key, status="running", started_at=now())

                def progress(value, message, report=None):
                    fields = dict(progress=value, message=message)
                    if report is not None:
                        fields["report"] = report
                    store.update(table, key, **fields)

                result = worker(key, event.is_set, progress)
                if event.is_set():
                    raise engine.Cancelled()
                store.update(
                    table,
                    key,
                    status="succeeded",
                    progress=100,
                    report=result,
                    message="已完成。",
                    finished_at=now(),
                )
            except engine.Cancelled:
                store.update(
                    table,
                    key,
                    status="cancelled",
                    message="已停止；可从已完成批次或 epoch 恢复。",
                    finished_at=now(),
                )
            except Exception as exc:
                store.update(
                    table, key, status="failed", error=str(exc), finished_at=now()
                )
            finally:
                self.cancellations.pop(key, None)

        self.pool.submit(run)
        return body

    def cancel(self, table, key):
        item = require(table, key)
        event = self.cancellations.get(key)
        if event:
            event.set()
            return store.update(table, key, status="cancelling")
        return item

    def can_resume(self, table, key):
        item = require(table, key)
        if (
            item["status"] not in ("failed", "cancelled", "interrupted")
            or key in self.cancellations
        ):
            raise HTTPException(409, "仅失败、停止或中断任务可以恢复。")
        return item


def assert_unused(key):
    for table in (
        "experiments",
        "batch_runs",
        "calibrations",
        "defense_runs",
        "training_runs",
    ):
        for job in store.all_items(table):
            if job["status"] in ("queued", "running", "cancelling"):
                import json

                if key in json.dumps(job, ensure_ascii=False):
                    raise HTTPException(409, "资源正在被任务使用，请先停止任务。")
    for session in store.all_items("guard_sessions"):
        if session.get("active"):
            profile = store.get("defenses", session["defense_id"])
            if key == session["defense_id"] or profile and key in profile["model_ids"]:
                raise HTTPException(409, "资源正在被防护会话使用，请先停止该会话。")
