"""Real PyTorch inference and bounded FGSM/PGD. No fabricated evaluation data."""

import hashlib
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models

torch.set_num_threads(min(4, torch.get_num_threads()))


def make_cifar_resnet18(**options):
    """32×32 图像使用小卷积和无最大池化的 ResNet-18。"""
    model = models.resnet18(**options)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model


def make_cifar_resnet34(**options):
    model = models.resnet34(**options)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model


ARCHITECTURES = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "mobilenet_v2": models.mobilenet_v2,
    "cifar_resnet18": make_cifar_resnet18,
    "cifar_resnet34": make_cifar_resnet34,
    "densenet121": models.densenet121,
    "efficientnet_b0": models.efficientnet_b0,
    "wide_resnet50_2": models.wide_resnet50_2,
}


def select_device():
    """默认要求 CUDA。只有显式设置 AEGIS_DEVICE=cpu 才允许 CPU 测试。"""
    device_name = os.environ.get("AEGIS_DEVICE", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            "当前环境未检测到 CUDA。请在 GPU 服务器安装匹配的 PyTorch；不会自动退回 CPU。"
        )
    return torch.device(device_name)


class Cancelled(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_model(spec):
    """先构造网络，再加载 state_dict；模型结构和预处理配置均由用户明确提供。"""
    device = select_device()
    if spec["architecture"] == "custom":
        path = Path(spec["loader_path"]).expanduser().resolve()
        if not path.is_file() or path.suffix != ".py":
            raise ValueError("自定义加载文件必须是本机存在的 .py 文件。")
        module_spec = importlib.util.spec_from_file_location(
            "aegis_custom_" + spec["id"].replace("-", ""), path
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        model = getattr(module, spec["factory"])()
    else:
        model = ARCHITECTURES[spec["architecture"]](
            weights=None, num_classes=spec["num_classes"]
        )
    if not isinstance(model, torch.nn.Module):
        raise ValueError("加载函数必须返回 torch.nn.Module。")
    path = Path(spec["weights_path"]).expanduser().resolve()
    if not path.is_file():
        raise ValueError("找不到权重文件，请使用运行后端的机器上的绝对路径。")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if (
        isinstance(state, dict)
        and state
        and all(str(k).startswith("module.") for k in state)
    ):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, device


def normalize(x, spec):
    mean = x.new_tensor(spec["mean"]).view(1, -1, 1, 1)
    std = x.new_tensor(spec["std"]).view(1, -1, 1, 1)
    return (x - mean) / std


def forward(model, x, spec):
    logits = model(normalize(x, spec))
    if (
        not torch.is_tensor(logits)
        or logits.ndim != 2
        or logits.shape[1] != spec["num_classes"]
    ):
        raise ValueError("模型必须输出 [batch, num_classes] 的 logits 张量。")
    if not torch.isfinite(logits).all():
        raise ValueError("模型输出包含 NaN 或 Inf。")
    return logits


def validate_model(spec):
    t = time.perf_counter()
    model, device = load_model(spec)
    x = torch.full(
        (1, 3, spec["input_size"], spec["input_size"]),
        0.5,
        device=device,
        requires_grad=True,
    )
    logits = forward(model, x, spec)
    grad = torch.autograd.grad(logits.max(), x)[0]
    if not torch.isfinite(grad).all():
        raise ValueError("输入梯度包含 NaN 或 Inf。")
    return {
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "sha256": sha256(spec["weights_path"]),
        "gradient_available": True,
        "loader_sha256": (
            sha256(spec["loader_path"]) if spec["architecture"] == "custom" else None
        ),
        "validation_seconds": round(time.perf_counter() - t, 3),
    }


def attack_tensor(
    model,
    x,
    y,
    spec,
    algorithm,
    epsilon,
    steps,
    step_size,
    seed,
    callback=None,
    cancelled=None,
    record_trace=True,
):
    # Bounds and epsilon always refer to the RGB input in [0,1], before normalization.
    generator = torch.Generator(device=x.device).manual_seed(seed)
    if algorithm == "fgsm":
        current = x.clone()
        iterations = 1
        alpha = epsilon
    elif algorithm == "pgd":
        noise = torch.rand(x.shape, device=x.device, generator=generator) * 2 - 1
        current = (x + noise * epsilon).clamp(0, 1)
        iterations = steps
        alpha = step_size
    else:
        raise ValueError("不支持的攻击。")
    trace = []
    for i in range(iterations):
        if cancelled and cancelled():
            raise Cancelled()
        current = current.detach().requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(forward(model, current, spec), y)
        grad = torch.autograd.grad(loss, current)[0]
        if not torch.isfinite(grad).all():
            raise ValueError("攻击梯度异常。")
        current = current.detach() + alpha * grad.sign()
        current = (
            torch.maximum(torch.minimum(current, x + epsilon), x - epsilon)
            .clamp(0, 1)
            .detach()
        )
        if record_trace:
            with torch.no_grad():
                probs = forward(model, current, spec).softmax(-1)
                target_probability = float(probs.gather(1, y[:, None]).mean())
            trace.append(
                {
                    "step": i + 1,
                    "target_probability": target_probability,
                    "loss_before_update": float(loss.detach()),
                }
            )
        if callback:
            callback(i + 1, iterations, trace)
    return current, trace


def image_tensor(path, size, device):
    with Image.open(path) as image:
        # Explicit, recorded square resize; no silent crop of the user image.
        array = (
            np.asarray(
                image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ).copy()
            / 255
        )
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def save_tensor(x, path):
    arr = x.detach()[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.round(np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def top_predictions(logits, categories):
    values, indices = logits.softmax(-1)[0].topk(min(5, logits.shape[1]))
    return [
        {
            "index": int(i),
            "label": categories[int(i)] if categories else f"类别 {int(i)}",
            "probability": float(v),
        }
        for v, i in zip(values, indices)
    ]


def evaluate_image(spec, task, directory, callback=None, cancelled=None):
    started = time.perf_counter()
    model, device = load_model(spec)
    x = image_tensor(directory / "input.png", spec["input_size"], device)
    with torch.no_grad():
        clean = forward(model, x, spec)
    clean_class = int(clean.argmax(-1).item())
    label = task["true_label"] if task["true_label"] is not None else clean_class
    y = torch.tensor([label], device=device)
    from .attacks import run_attack
    from .attack_protocol import AttackConfig

    options = AttackConfig(
        **{key: task[key] for key in AttackConfig.model_fields if key in task}
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    adv, details = run_attack(
        model,
        x,
        y,
        spec,
        options,
        callback,
        cancelled,
        log_path=(
            directory / "autoattack.log" if options.algorithm == "autoattack" else None
        ),
    )
    trace = details["trace"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with torch.no_grad():
        attacked = forward(model, adv, spec)
    adv_class = int(attacked.argmax(-1).item())
    save_tensor(x, directory / "original.png")
    save_tensor(adv, directory / "adversarial.png")
    delta = adv - x
    # Neutral gray = no perturbation, with display gain recorded separately.
    gain = 16
    save_tensor(0.5 + gain * delta, directory / "perturbation.png")
    np.save(directory / "adversarial_float32.npy", adv.detach().cpu().numpy())
    np.save(directory / "perturbation_float32.npy", delta.detach().cpu().numpy())
    # PNG previews are quantized; quantitative metrics refer to the saved float tensors.
    originally_correct = None if task["true_label"] is None else clean_class == label
    return {
        "attack_protocol": details["protocol"],
        "dependencies": details["dependencies"],
        "clean_top5": top_predictions(clean, spec["categories"]),
        "adversarial_top5": top_predictions(attacked, spec["categories"]),
        "prediction_changed": clean_class != adv_class,
        "attack_success": (
            None if originally_correct is not True else adv_class != label
        ),
        "originally_correct": originally_correct,
        "true_label": task["true_label"],
        "attack_label": label,
        "label_source": (
            "ground_truth" if task["true_label"] is not None else "original_prediction"
        ),
        "linf": float(delta.abs().max()),
        "l2": float(delta.flatten().norm()),
        "seconds": round(time.perf_counter() - started, 3),
        "device": str(device),
        "trace": trace,
        "perturbation_display_gain": gain,
        "preprocessing": {
            "resize": "bilinear_square",
            "size": spec["input_size"],
            "mean": spec["mean"],
            "std": spec["std"],
        },
        "note": "指标使用浮点张量计算；PNG 用于图片预览。",
    }
