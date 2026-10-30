"""防御配置、校准任务和攻防评测接口。"""

import io
import json
import uuid
import numpy as np
import torch
from fastapi import HTTPException, Form, UploadFile, File
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from . import store, engine
from .defenses import DefenseConfig, DefenseRuntime, TransformConfig, transform_images
from .defense_evaluation import (
    DefenseEvaluationConfig,
    calibrate,
    run_defense_suite,
    calibration_signature,
)
from .jobs import JobQueue, require, assert_unused, now
from .uploads import read_image


class CalibrationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    defense_id: str
    dataset_id: str
    target_fpr: float = Field(default=0.05, gt=0, lt=0.5)
    sample_limit: int = Field(default=0, ge=0, le=1000000)
    batch_size: int = Field(default=32, ge=1, le=512)


def profile_members(profile):
    return [require("models", key) for key in profile["model_ids"]]


def check_profile_contract(config):
    specs = [require("models", key) for key in config.model_ids]
    for spec in specs:
        if any(
            spec.get(key, []) != specs[0].get(key, [])
            for key in ("num_classes", "input_size", "categories")
        ):
            raise HTTPException(422, "成员的输入尺寸、类别数和类别顺序必须相同。")
    return specs


def install_workflows(app, pool, cancellations):
    queue = JobQueue(pool, cancellations)

    @app.get("/api/defenses")
    def profiles():
        return [
            item for item in store.all_items("defenses") if not item.get("archived")
        ]

    @app.post("/api/defenses", status_code=201)
    def create_profile(config: DefenseConfig):
        check_profile_contract(config)
        return store.save(
            "defenses",
            dict(
                id=str(uuid.uuid4()),
                config=config.model_dump(),
                name=config.name,
                model_ids=config.model_ids,
                created_at=now(),
                status="configured",
                calibration=None,
            ),
        )

    @app.put("/api/defenses/{key}")
    def edit_profile(key: str, config: DefenseConfig):
        require("defenses", key)
        assert_unused(key)
        check_profile_contract(config)
        return store.update(
            "defenses",
            key,
            config=config.model_dump(),
            name=config.name,
            model_ids=config.model_ids,
            status="configured",
            calibration=None,
            updated_at=now(),
        )

    @app.post("/api/defenses/{key}/validate")
    def validate_profile(key: str):
        profile = require("defenses", key)
        specs = profile_members(profile)

        def compute():
            runtime = DefenseRuntime(profile["config"], specs)
            images = torch.full(
                (1, 3, specs[0]["input_size"], specs[0]["input_size"]),
                0.5,
                device=runtime.device,
            )
            measured = runtime.measure(images)
            return dict(
                protocol=runtime.protocol(),
                prediction=int(measured["prediction"][0]),
                resource_signature=calibration_signature(profile["config"], specs),
            )

        try:
            result = pool.submit(compute).result()
            return store.update(
                "defenses", key, status="validated", validation=result, error=None
            )
        except Exception as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/transform-preview")
    async def preview(config: str = Form(...), image: UploadFile = File(...)):
        original, _ = await read_image(image)
        try:
            transforms = [
                TransformConfig.model_validate(item) for item in json.loads(config)
            ]
            if len(transforms) > 8:
                raise ValueError("最多八个处理步骤。")
            # 预览是本地图片处理，不加载模型；缩小过大图片避免滤波预览占用过多内存。
            original.thumbnail((512, 512))
            if min(original.size) < 16:
                raise ValueError("预览图片宽高至少为 16。")
            array = np.asarray(original, dtype=np.float32).copy() / 255
            tensor = torch.from_numpy(array).permute(2, 0, 1)[None]
            processed = transform_images(
                tensor, transforms, torch.Generator().manual_seed(0)
            )
            from PIL import Image

            buffer = io.BytesIO()
            Image.fromarray(
                np.round(processed[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            ).save(buffer, format="PNG")
            return Response(buffer.getvalue(), media_type="image/png")
        except Exception as exc:
            raise HTTPException(422, str(exc)) from exc

    def calibration_worker(body):
        def worker(key, cancelled, progress):
            cfg = CalibrationInput.model_validate(body["config"])
            profile = require("defenses", cfg.defense_id)
            dataset = require("datasets", cfg.dataset_id)
            return calibrate(
                profile,
                profile_members(profile),
                dataset,
                store.DATA / key,
                cfg.target_fpr,
                cfg.sample_limit,
                cfg.batch_size,
                cancelled,
                progress,
            )

        return worker

    @app.get("/api/calibrations")
    def calibration_list():
        return store.all_items("calibrations")

    @app.post("/api/calibrations", status_code=202)
    def create_calibration(config: CalibrationInput):
        profile = require("defenses", config.defense_id)
        dataset = require("datasets", config.dataset_id)
        if dataset.get("status") != "validated" or dataset["split"] != "validation":
            raise HTTPException(
                422, "请先登记并检查独立 validation 数据集；不允许使用测试集校准。"
            )
        body = dict(config=config.model_dump(), name=profile["name"] + " · 阈值校准")
        return queue.submit("calibrations", body, calibration_worker(body))

    @app.post("/api/calibrations/{key}/apply")
    def apply_calibration(key: str):
        job = require("calibrations", key)
        if job["status"] != "succeeded":
            raise HTTPException(409, "校准尚未成功。")
        result = job["report"]
        profile = require("defenses", job["config"]["defense_id"])
        assert_unused(profile["id"])
        specs = profile_members(profile)
        current_signature = calibration_signature(profile["config"], specs)
        existing = profile.get("calibration")
        if (
            existing
            and existing.get("job_id") == key
            and existing["applied_signature"] == current_signature
        ):
            return profile
        if (
            calibration_signature(profile["config"], specs)
            != result["profile_signature"]
        ):
            raise HTTPException(409, "防御或模型已改变，不能应用旧阈值。")
        config = DefenseConfig.model_validate(profile["config"])
        config.detector.threshold = max(0, min(1, result["threshold"]))
        config = config.model_dump()
        calibration = dict(
            result, job_id=key, applied_signature=calibration_signature(config, specs)
        )
        return store.update(
            "defenses",
            profile["id"],
            config=config,
            calibration=calibration,
            updated_at=now(),
        )

    def evaluation_worker(body):
        def worker(key, cancelled, progress):
            cfg = DefenseEvaluationConfig.model_validate(body["config"])
            profiles = [require("defenses", item) for item in cfg.defense_ids]
            members = {item["id"]: profile_members(item) for item in profiles}
            return run_defense_suite(
                cfg,
                require("datasets", cfg.dataset_id),
                profiles,
                members,
                store.DATA / key,
                progress,
                cancelled,
            )

        return worker

    @app.get("/api/defense-runs")
    def evaluation_list():
        return [
            {key: value for key, value in item.items() if key != "report"}
            for item in store.all_items("defense_runs")
        ]

    @app.post("/api/defense-runs", status_code=202)
    def create_evaluation(config: DefenseEvaluationConfig):
        profiles = [require("defenses", key) for key in config.defense_ids]
        dataset = require("datasets", config.dataset_id)
        if dataset.get("status") != "validated":
            raise HTTPException(422, "请先检查数据集。")
        body = dict(
            config=config.model_dump(),
            name=" / ".join(item["name"] for item in profiles),
            dataset_name=dataset["name"],
        )
        return queue.submit("defense_runs", body, evaluation_worker(body))

    @app.get("/api/workflows/{kind}/{key}")
    def job_detail(kind: str, key: str):
        if kind not in ("calibrations", "defense_runs", "training_runs"):
            raise HTTPException(404, "任务类型不存在。")
        return require(kind, key)

    @app.post("/api/workflows/{kind}/{key}/cancel")
    def cancel_job(kind: str, key: str):
        if kind not in ("calibrations", "defense_runs", "training_runs"):
            raise HTTPException(404, "任务类型不存在。")
        return queue.cancel(kind, key)

    @app.post("/api/workflows/{kind}/{key}/resume", status_code=202)
    def resume_job(kind: str, key: str):
        if kind not in ("calibrations", "defense_runs"):
            raise HTTPException(404, "训练任务使用其独立恢复接口。")
        body = queue.can_resume(kind, key)
        return queue.submit(
            kind,
            body,
            (calibration_worker if kind == "calibrations" else evaluation_worker)(body),
            resume=True,
        )

    @app.get("/api/workflows/{kind}/{key}/files/{filename:path}")
    def download(kind: str, key: str, filename: str):
        if kind not in ("calibrations", "defense_runs", "training_runs"):
            raise HTTPException(404, "任务类型不存在。")
        require(kind, key)
        directory = (store.DATA / key).resolve()
        path = (directory / filename).resolve()
        if (
            not path.is_relative_to(directory)
            or path.suffix
            not in (".json", ".csv", ".png", ".html", ".log", ".jsonl", ".pt")
            or not path.is_file()
        ):
            raise HTTPException(404, "文件不存在。")
        return FileResponse(path, filename=path.name)
