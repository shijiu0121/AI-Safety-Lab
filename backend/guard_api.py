"""图像分类接口的实时推理防护。"""

import asyncio
import time
import uuid
from pathlib import Path
import numpy as np
import torch
from fastapi import HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from . import store, engine
from .defenses import DefenseRuntime, transform_images
from .defense_evaluation import calibration_signature
from .workflow_api import profile_members
from .jobs import require, now
from .uploads import read_image


class GuardInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    defense_id: str
    policy: Literal["monitor", "reject"] = "monitor"
    save_images: bool = False


def install_guard(app, pool, cancellations):
    @app.get("/api/guard/sessions")
    def sessions():
        return store.all_items("guard_sessions")

    @app.post("/api/guard/sessions", status_code=201)
    def create_session(config: GuardInput):
        require("defenses", config.defense_id)
        return store.save(
            "guard_sessions",
            dict(
                config.model_dump(), id=str(uuid.uuid4()), active=True, created_at=now()
            ),
        )

    @app.post("/api/guard/sessions/{key}/stop")
    def stop_session(key: str):
        require("guard_sessions", key)
        return store.update("guard_sessions", key, active=False, stopped_at=now())

    @app.get("/api/guard/events")
    def events(session_id: str = "", limit: int = 100):
        selected = [
            item
            for item in store.all_items("guard_events")
            if not session_id or item["session_id"] == session_id
        ]
        recent = selected[: max(1, min(limit, 500))]
        successful = [item for item in selected if item["status"] == "succeeded"]
        latencies = [item["seconds"] for item in successful]
        return dict(
            events=recent,
            scope="当前会话全部已保存事件；列表只返回最近记录",
            total=len(selected),
            rejected=sum(item.get("decision") == "reject" for item in selected),
            flagged=sum(item.get("flagged", False) for item in selected),
            errors=sum(item["status"] == "failed" for item in selected),
            mean_seconds=sum(latencies) / len(latencies) if latencies else None,
            p95_seconds=float(np.quantile(latencies, 0.95)) if latencies else None,
        )

    @app.post("/api/guard/sessions/{key}/predict")
    async def predict(key: str, image: UploadFile = File(...)):
        session = require("guard_sessions", key)
        if not session["active"]:
            raise HTTPException(409, "该防护会话已停止。")
        if cancellations:
            raise HTTPException(429, "GPU 正在运行其他任务，请稍后再试。")
        event_id = str(uuid.uuid4())
        started = time.perf_counter()
        body = dict(
            id=event_id,
            session_id=key,
            created_at=now(),
            status="running",
            original_filename=Path(image.filename or "image").name,
        )
        try:
            original, information = await read_image(image)
        except HTTPException as exc:
            store.save(
                "guard_events",
                dict(
                    body,
                    status="failed",
                    error=str(exc.detail),
                    decision="reject",
                    flagged=True,
                    seconds=round(time.perf_counter() - started, 4),
                ),
            )
            raise
        import threading

        if cancellations:
            raise HTTPException(429, "GPU 正在运行其他任务，请稍后重试。")
        cancellations[event_id] = threading.Event()

        def compute():
            profile = require("defenses", session["defense_id"])
            specs = profile_members(profile)
            signature = calibration_signature(profile["config"], specs)
            if (
                profile.get("calibration")
                and profile["calibration"]["applied_signature"] != signature
            ):
                raise ValueError("校准对应的模型/配置已改变，请重新校准。")
            runtime = DefenseRuntime(profile["config"], specs)
            size = specs[0]["input_size"]
            from PIL import Image

            array = (
                np.asarray(
                    original.resize((size, size), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                ).copy()
                / 255
            )
            tensor = torch.from_numpy(array).permute(2, 0, 1)[None].to(runtime.device)
            request_seed = (runtime.config.seed + int(uuid.UUID(event_id))) % 2000000000
            measured = runtime.measure(tensor, request_seed)
            if runtime.device.type == "cuda":
                torch.cuda.synchronize(runtime.device)
            flag = bool(measured["flagged"][0])
            decision = "reject" if flag and session["policy"] == "reject" else "allow"
            reasons = []
            if bool(measured["detected"][0]):
                reasons.append("output_detector")
            if bool(measured["anomaly"][0]):
                reasons.append("input_statistics")
            result = dict(
                body,
                status="succeeded",
                information=information,
                flagged=flag,
                reasons=reasons,
                decision=decision,
                prediction=(
                    int(measured["prediction"][0]) if decision == "allow" else None
                ),
                audit_prediction=int(measured["prediction"][0]),
                confidence=float(measured["confidence"][0]),
                score=float(measured["score"][0]),
                entropy=float(measured["entropy"][0]),
                input_std=float(measured["input_std"][0]),
                saturation=float(measured["saturation"][0]),
                seconds=round(time.perf_counter() - started, 4),
                protocol=runtime.protocol(),
                resource_signature=signature,
                request_seed=request_seed,
                latency_note="含图片读取、模型加载、检测和推理。",
            )
            if session["save_images"]:
                folder = store.DATA / "guard" / event_id
                folder.mkdir(parents=True, exist_ok=True)
                original.save(folder / "original.png")
                processed = transform_images(
                    tensor,
                    runtime.config.transforms,
                    torch.Generator(device=runtime.device).manual_seed(
                        runtime.config.seed
                    ),
                )
                engine.save_tensor(processed, folder / "processed.png")
                result["images_saved"] = True
            return store.save("guard_events", result)

        try:
            return await asyncio.wrap_future(pool.submit(compute))
        except Exception as exc:
            store.save(
                "guard_events",
                dict(
                    body,
                    status="failed",
                    error=str(exc),
                    decision="reject",
                    flagged=True,
                    seconds=round(time.perf_counter() - started, 4),
                ),
            )
            raise HTTPException(422, str(exc)) from exc
        finally:
            cancellations.pop(event_id, None)

    @app.get("/api/guard/events/{key}/images/{kind}")
    def event_image(key: str, kind: Literal["original", "processed"]):
        item = require("guard_events", key)
        path = store.DATA / "guard" / key / f"{kind}.png"
        if not item.get("images_saved") or not path.is_file():
            raise HTTPException(404, "该事件未保存图片。")
        return FileResponse(path)
