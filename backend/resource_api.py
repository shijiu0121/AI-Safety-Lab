"""资源 JSON 导入、修改、导出和可恢复归档；不删除用户权重或数据。"""

import json
import uuid
from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict
from . import store, engine
from .model_schema import ModelInput, model_payload
from .data_sources import DatasetInput
from .jobs import require, assert_unused, now


class ImportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    resource: dict


def install_resources(app):
    @app.get("/api/workspace-summary")
    def workspace_summary():
        counts = {}
        for kind in (
            "experiments",
            "batch_runs",
            "calibrations",
            "defense_runs",
            "training_runs",
        ):
            items = store.all_items(kind)
            counts[kind] = dict(
                total=len(items),
                completed=sum(item["status"] == "succeeded" for item in items),
                running=sum(
                    item["status"] in ("queued", "running", "cancelling")
                    for item in items
                ),
            )
        return dict(
            kinds=counts,
            completed=sum(item["completed"] for item in counts.values()),
            running=sum(item["running"] for item in counts.values()),
        )

    @app.get("/api/catalog")
    def catalog():
        return dict(
            architectures=list(engine.ARCHITECTURES) + ["custom"],
            training_methods=["erm", "pgd_at", "trades", "at_bsl", "taet", "distill_teacher", "distill"],
            transforms=["bit_depth", "median", "gaussian", "resize_pad", "noise"],
            detectors=["none", "confidence", "entropy", "consistency"],
        )

    @app.post("/api/resources/import", status_code=201)
    def import_resource(payload: ImportInput):
        try:
            if payload.kind == "models":
                body = model_payload(payload.resource)
                fields = dict(status="registered", validation=None)
            elif payload.kind == "datasets":
                allowed = set(DatasetInput.model_fields)
                unknown = (
                    set(payload.resource)
                    - allowed
                    - {
                        "id",
                        "status",
                        "created_at",
                        "updated_at",
                        "inspection",
                        "error",
                        "archived",
                    }
                )
                if unknown:
                    raise ValueError(f"未知数据集字段：{sorted(unknown)}")
                body = DatasetInput.model_validate(
                    {
                        key: value
                        for key, value in payload.resource.items()
                        if key in allowed
                    }
                ).model_dump()
                fields = dict(status="registered", inspection=None)
            else:
                raise ValueError("kind 只接受 models 或 datasets。")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return store.save(
            payload.kind, dict(body, id=str(uuid.uuid4()), created_at=now(), **fields)
        )

    @app.put("/api/models/{key}")
    def edit_model(key: str, body: ModelInput):
        require("models", key)
        assert_unused(key)
        return store.update(
            "models",
            key,
            **body.model_dump(),
            status="registered",
            validation=None,
            error=None,
            updated_at=now(),
        )

    @app.put("/api/datasets/{key}")
    def edit_dataset(key: str, body: DatasetInput):
        require("datasets", key)
        assert_unused(key)
        return store.update(
            "datasets",
            key,
            **body.model_dump(),
            status="registered",
            inspection=None,
            error=None,
            updated_at=now(),
        )

    @app.get("/api/resources/{kind}/{key}/export")
    def export_resource(kind: str, key: str):
        if kind not in ("models", "datasets", "defenses"):
            raise HTTPException(404, "未知资源类型。")
        item = require(kind, key)
        body = (
            item["config"]
            if kind == "defenses"
            else {
                field: item[field]
                for field in (
                    ModelInput.model_fields
                    if kind == "models"
                    else DatasetInput.model_fields
                )
                if field in item
            }
        )
        return Response(
            json.dumps(body, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{kind}-{key}.json"'
            },
        )

    @app.post("/api/resources/{kind}/{key}/archive")
    def archive(kind: str, key: str):
        if kind not in ("models", "datasets", "defenses"):
            raise HTTPException(404, "未知资源类型。")
        require(kind, key)
        assert_unused(key)
        if kind == "models" and any(
            not item.get("archived") and key in item["model_ids"]
            for item in store.all_items("defenses")
        ):
            raise HTTPException(409, "模型仍属于防御配置，请先修改或归档对应防御。")
        return store.update(kind, key, archived=True, updated_at=now())

    @app.get("/api/resources/archived")
    def archived():
        return [
            dict(item, kind=kind)
            for kind in ("models", "datasets", "defenses")
            for item in store.all_items(kind)
            if item.get("archived")
        ]

    @app.post("/api/resources/{kind}/{key}/restore")
    def restore(kind: str, key: str):
        if (
            kind not in ("models", "datasets", "defenses")
            or store.get(kind, key) is None
        ):
            raise HTTPException(404, "资源不存在。")
        return store.update(kind, key, archived=False, updated_at=now())
