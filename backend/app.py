import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import torchvision
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, model_validator

from . import engine, store
from .attack_protocol import AttackConfig, check_attack_compatibility

POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aegis-compute")
CANCEL = {}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def now():
    return datetime.now(timezone.utc).isoformat()


from .model_schema import ModelInput


class AttackInput(AttackConfig):
    model_id: str
    true_label: int | None = Field(default=None, ge=0)


@asynccontextmanager
async def lifespan(app):
    store.initialize()
    yield
    for event in list(CANCEL.values()):
        event.set()


app = FastAPI(title="Aegis Lab", version="0.4.0", lifespan=lifespan)


@app.get("/api/health")
def health():
    example = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    requested_device = os.environ.get("AEGIS_DEVICE", "cuda")
    ready = not requested_device.startswith("cuda") or torch.cuda.is_available()
    return {
        "status": "ok",
        "phase": "03",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": requested_device,
        "compute_ready": ready,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "cached_example_available": example.is_file(),
        "features": [
            "model_registration",
            "model_validation",
            "fgsm",
            "pgd",
            "single_image_experiments",
            "pgd20",
            "pgd100",
            "cw_linf",
            "deepfool",
            "autoattack_standard",
            "dataset_registry",
            "batch_evaluation",
            "checkpoint_resume",
            "task_persistence",
            "task_cancellation",
            "preprocessing_defenses",
            "detectors",
            "ensemble_defenses",
            "validation_calibration",
            "adaptive_bpda_eot",
            "defense_evaluation",
            "training_center",
            "distillation",
            "input_output_anomaly_rules",
            "realtime_inference_guard",
            "resource_management",
            "html_reports",
        ],
    }


@app.get("/api/models")
def models_list():
    return [item for item in store.all_items("models") if not item.get("archived")]


@app.post("/api/models", status_code=201)
def create_model(spec: ModelInput):
    body = dict(
        spec.model_dump(),
        id=str(uuid.uuid4()),
        created_at=now(),
        status="registered",
        validation=None,
    )
    return store.save("models", body)


@app.post("/api/models/import-cached-example", status_code=201)
def import_example():
    path = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    if not path.is_file():
        raise HTTPException(
            404, "未发现本机缓存权重，请自行接入模型；本功能不会自动下载。"
        )
    for item in store.all_items("models"):
        if item["weights_path"] == str(path):
            return item
    spec = ModelInput(
        name="ResNet-18 · ImageNet / 本机缓存",
        architecture="resnet18",
        weights_path=str(path),
        categories=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"],
    )
    return create_model(spec)


@app.post("/api/models/{model_id}/validate")
def validate(model_id: str):
    spec = store.get("models", model_id)
    if not spec:
        raise HTTPException(404, "模型不存在。")
    try:
        validation = POOL.submit(engine.validate_model, spec).result()
        return store.update(
            "models", model_id, status="validated", validation=validation, error=None
        )
    except Exception as exc:
        store.update("models", model_id, status="invalid", error=str(exc))
        raise HTTPException(422, str(exc)) from exc


def run_experiment(task_id):
    event = CANCEL[task_id]
    try:
        task = store.get("experiments", task_id)
        if event.is_set():
            store.update("experiments", task_id, status="cancelled", finished_at=now())
            return
        store.update(
            "experiments", task_id, status="running", started_at=now(), progress=0
        )
        spec = store.get("models", task["model_id"])
        # Freeze the complete model contract, including weight digest, with this run.
        snapshot = dict(spec)
        digest = engine.sha256(spec["weights_path"])
        snapshot["weights_sha256"] = digest
        if digest != spec["validation"]["sha256"]:
            raise ValueError("权重在验证后已变化，请重新验证模型。")
        if spec["architecture"] == "custom":
            snapshot["loader_sha256"] = engine.sha256(spec["loader_path"])
            if snapshot["loader_sha256"] != spec["validation"].get("loader_sha256"):
                raise ValueError("自定义加载文件发生变化或尚未校验，请重新验证模型。")
        store.update("experiments", task_id, model_snapshot=snapshot)

        def progress(step, total, trace):
            store.update(
                "experiments",
                task_id,
                progress=round(step / total * 99),
                step=step,
                trace=trace,
            )

        result = engine.evaluate_image(
            spec, task, store.DATA / task_id, progress, event.is_set
        )
        if event.is_set():
            raise engine.Cancelled()
        store.update(
            "experiments",
            task_id,
            status="succeeded",
            progress=100,
            result=result,
            finished_at=now(),
        )
    except engine.Cancelled:
        store.update("experiments", task_id, status="cancelled", finished_at=now())
    except Exception as exc:
        store.update(
            "experiments", task_id, status="failed", error=str(exc), finished_at=now()
        )
    finally:
        CANCEL.pop(task_id, None)


@app.post("/api/experiments", status_code=202)
async def create_experiment(config: str = Form(...), image: UploadFile = File(...)):
    try:
        cfg = AttackInput.model_validate_json(config)
    except Exception as exc:
        raise HTTPException(422, "实验配置无效：" + str(exc)) from exc
    spec = store.get("models", cfg.model_id)
    if not spec or spec["status"] != "validated":
        raise HTTPException(422, "请先接入并成功验证模型。")
    if cfg.true_label is not None and cfg.true_label >= spec["num_classes"]:
        raise HTTPException(422, "真实类别编号超出模型输出范围。")
    try:
        check_attack_compatibility(cfg, spec["num_classes"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if len(CANCEL) >= 8:
        raise HTTPException(429, "当前排队任务较多，请稍后再试。")
    payload = await image.read(12 * 1024 * 1024 + 1)
    if len(payload) > 12 * 1024 * 1024:
        raise HTTPException(413, "图片不能超过 12 MB。")
    import io

    try:
        with Image.open(io.BytesIO(payload)) as im:
            if im.width * im.height > 16_000_000:
                raise ValueError("图片不能超过 1600 万像素。")
            im.load()
            original = im.convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(422, "无法读取图片：" + str(exc)) from exc
    key = str(uuid.uuid4())
    directory = store.DATA / key
    directory.mkdir()
    original.save(directory / "input.png")
    body = dict(
        cfg.model_dump(),
        attack_protocol=cfg.protocol(),
        id=key,
        created_at=now(),
        model_name=spec["name"],
        status="queued",
        progress=0,
        original_filename=Path(image.filename or "image").name,
        result=None,
        error=None,
        input_sha256=engine.sha256(directory / "input.png"),
        runtime={"torch": torch.__version__, "torchvision": torchvision.__version__},
    )
    store.save("experiments", body)
    CANCEL[key] = threading.Event()
    POOL.submit(run_experiment, key)
    return body


@app.get("/api/experiments")
def list_experiments():
    return store.all_items("experiments")


@app.get("/api/experiments/{key}")
def experiment(key: str):
    item = store.get("experiments", key)
    if item is None:
        raise HTTPException(404, "实验不存在。")
    return item


@app.post("/api/experiments/{key}/cancel")
def cancel(key: str):
    item = experiment(key)
    event = CANCEL.get(key)
    if event:
        event.set()
        return store.update("experiments", key, status="cancelling")
    return item


@app.get("/api/experiments/{key}/images/{kind}")
def experiment_image(
    key: str, kind: Literal["original", "adversarial", "perturbation"]
):
    experiment(key)
    path = store.DATA / key / (kind + ".png")
    if not path.is_file():
        raise HTTPException(404, "结果图片尚未生成。")
    return FileResponse(path)


@app.get("/api/experiments/{key}/report")
def report(key: str):
    item = experiment(key)
    from fastapi.responses import Response

    return Response(
        json.dumps(item, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="experiment-{key}.json"'
        },
    )


@app.get("/api/experiments/{key}/tensors/{kind}")
def tensor_download(key: str, kind: Literal["adversarial", "perturbation"]):
    experiment(key)
    path = store.DATA / key / (kind + "_float32.npy")
    if not path.is_file():
        raise HTTPException(404, "浮点结果尚未生成。")
    return FileResponse(path, filename=path.name)


from .batch_api import install_routes

install_routes(app, POOL, CANCEL)
from .resource_api import install_resources
from .workflow_api import install_workflows
from .training_api import install_training
from .guard_api import install_guard

install_resources(app)
install_workflows(app, POOL, CANCEL)
install_training(app, POOL, CANCEL)
install_guard(app, POOL, CANCEL)


DIST = store.ROOT / "dist"
if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/")
def index():
    if not (DIST / "index.html").is_file():
        raise HTTPException(503, "前端尚未构建，请执行 npm install 和 npm run build。")
    return FileResponse(DIST / "index.html")
