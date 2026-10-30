"""服务器环境自检。默认要求 CUDA；--cpu-only 仅检查本机构建，不能算 GPU 验收。"""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import torch
from .training_config import load_configuration


def check_environment(cpu_only=False, gpu_smoke=False, training_configs=()):
    result = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        gpu_test_executed=False,
        checks=[],
        training_configs=[],
    )
    for name in (
        "torchvision",
        "numpy",
        "Pillow",
        "fastapi",
        "uvicorn",
        "pydantic",
        "foolbox",
        "autoattack",
    ):
        try:
            result["checks"].append(
                dict(name=name, ok=True, version=importlib.metadata.version(name))
            )
        except importlib.metadata.PackageNotFoundError:
            result["checks"].append(
                dict(name=name, ok=False, error="missing dependency")
            )
    if not cpu_only:
        result["checks"].append(
            dict(name="CUDA required", ok=torch.cuda.is_available())
        )
    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["memory_bytes"] = dict(zip(("free", "total"), torch.cuda.mem_get_info()))
    if gpu_smoke:
        if cpu_only:
            raise ValueError("--gpu-smoke 与 --cpu-only 不能同时指定。")
        if torch.cuda.is_available():
            from .engine import make_cifar_resnet18

            model = make_cifar_resnet18(weights=None, num_classes=10).cuda().eval()
            images = torch.rand(2, 3, 32, 32, device="cuda", requires_grad=True)
            logits = model(images)
            gradient = torch.autograd.grad(logits.square().mean(), images)[0]
            torch.cuda.synchronize()
            result["gpu_test_executed"] = True
            result["checks"].append(
                dict(
                    name="CUDA forward/input-gradient smoke",
                    ok=bool(
                        torch.isfinite(logits).all() and torch.isfinite(gradient).all()
                    ),
                    note="随机网络，检查张量运行，不产生性能指标",
                )
            )
    for filename in training_configs:
        try:
            config = load_configuration(filename)
            data_exists = Path(config["data_dir"]).is_dir()
            result["training_configs"].append(
                dict(
                    path=str(filename),
                    schema_valid=True,
                    data_directory_exists=data_exists,
                    method=config["method"],
                    dataset=config["dataset"],
                )
            )
            result["checks"].append(dict(name=f"data path: {filename}", ok=data_exists))
        except Exception as exc:
            result["training_configs"].append(
                dict(path=str(filename), schema_valid=False, error=str(exc))
            )
            result["checks"].append(dict(name=f"config: {filename}", ok=False))
    result["ok"] = all(item["ok"] for item in result["checks"])
    result["scope"] = (
        "CPU build checks only"
        if cpu_only
        else "GPU environment; formal training and full attacks not performed"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Aegis Lab 部署自检")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--training-config", action="append", default=[])
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = check_environment(args.cpu_only, args.gpu_smoke, args.training_config)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
