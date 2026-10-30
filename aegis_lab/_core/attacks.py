"""统一攻击入口。CW 使用 L∞ 投影与 CW margin 损失；AA 调用固定官方版本。"""

import importlib.metadata
import json
import random
import threading
from contextlib import contextmanager

import numpy as np
import torch

from .attack_protocol import (
    AttackConfig,
    CATALOG,
    AUTOATTACK_COMMIT,
    check_attack_compatibility,
)
from .engine import Cancelled, forward

# 外部实现会使用全局随机状态。锁与状态恢复避免与其他任务互相污染。
RANDOM_LOCK = threading.RLock()


@contextmanager
def fixed_random_state(seed):
    with RANDOM_LOCK:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            try:
                yield
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)


class NormalizedClassifier(torch.nn.Module):
    """保留输入预处理的梯度，同时在每次前向计算前检查取消请求。"""

    def __init__(self, model, spec, cancelled=None):
        super().__init__()
        self.model = model
        self.spec = spec
        self.cancelled = cancelled

    def forward(self, images):
        if self.cancelled and self.cancelled():
            raise Cancelled()
        return forward(self.model, images, self.spec)


def margin_loss(logits, labels, confidence):
    """最大化 -max(z_true - max(z_other), -kappa)，不是经典 C&W-L2 优化器。"""
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    other_logits = (
        logits.masked_fill(
            torch.nn.functional.one_hot(labels, logits.shape[1]).bool(), -torch.inf
        )
        .max(1)
        .values
    )
    return -torch.clamp(true_logits - other_logits, min=-confidence)


def projected_attack(classifier, images, labels, config, callback=None):
    """保存迭代中已经找到的成功样本，多次重启取更强的逐样本结果。"""
    protocol = config.protocol()
    epsilon = protocol["budget"]
    iterations = protocol["iterations"]
    restarts = 1 if config.algorithm == "fgsm" else config.restarts
    generator = torch.Generator(device=images.device).manual_seed(config.seed)
    with torch.no_grad():
        clean_logits = classifier(images)
    best_images = images.clone()
    best_scores = torch.full_like(labels, -torch.inf, dtype=images.dtype)
    already_successful = clean_logits.argmax(1).ne(labels)
    trace = []
    for restart in range(restarts):
        if config.algorithm == "fgsm":
            current = images.clone()
        else:
            noise = (
                torch.rand(images.shape, device=images.device, generator=generator) * 2
                - 1
            )
            current = (images + epsilon * noise).clamp(0, 1)
        for iteration in range(iterations):
            current = current.detach().requires_grad_(True)
            logits = classifier(current)
            losses = (
                margin_loss(logits, labels, config.cw_confidence)
                if config.algorithm == "cw_linf"
                else torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            )
            gradient = torch.autograd.grad(losses.mean(), current)[0]
            if not torch.isfinite(gradient).all():
                raise ValueError("攻击梯度出现 NaN/Inf。")
            step_size = epsilon if config.algorithm == "fgsm" else protocol["step_size"]
            current = current.detach() + step_size * gradient.sign()
            current = (
                torch.maximum(
                    torch.minimum(current, images + epsilon), images - epsilon
                )
                .clamp(0, 1)
                .detach()
            )
            with torch.no_grad():
                updated_logits = classifier(current)
                scores = (
                    margin_loss(updated_logits, labels, config.cw_confidence)
                    if config.algorithm == "cw_linf"
                    else torch.nn.functional.cross_entropy(
                        updated_logits, labels, reduction="none"
                    )
                )
                successful = updated_logits.argmax(1).ne(labels)
                # 成功样本优先于尚未成功的样本，避免之后迭代又跨回正确类别。
                keep = (successful & ~already_successful) | (
                    (successful == already_successful) & (scores > best_scores)
                )
                best_images = torch.where(
                    keep[:, None, None, None], current, best_images
                )
                best_scores = torch.where(keep, scores, best_scores)
                already_successful |= successful
                probability = float(
                    updated_logits.softmax(1).gather(1, labels[:, None]).mean()
                )
            trace.append(
                {
                    "step": restart * iterations + iteration + 1,
                    "restart": restart + 1,
                    "target_probability": probability,
                }
            )
            if callback:
                callback(len(trace), restarts * iterations, trace)
    return best_images.detach(), trace


def package_versions():
    versions = {"torch": torch.__version__}
    for package in (
        "foolbox",
        "eagerpy",
        "autoattack",
        "torchvision",
        "numpy",
        "Pillow",
        "scipy",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        metadata = json.loads(
            importlib.metadata.distribution("autoattack").read_text("direct_url.json")
            or "{}"
        )
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        metadata = {}
    url = metadata.get("url", "")
    verified = url.startswith("https://github.com/fra31/auto-attack") and (
        AUTOATTACK_COMMIT in url
        or metadata.get("vcs_info", {}).get("commit_id") == AUTOATTACK_COMMIT
    )
    versions["autoattack_commit_verified"] = verified
    versions["autoattack_source"] = url if verified else "unverified"
    return versions


def run_attack(
    model, images, labels, spec, config, callback=None, cancelled=None, log_path=None
):
    if not isinstance(config, AttackConfig):
        config = AttackConfig.model_validate(config)
    check_attack_compatibility(config, spec["num_classes"])
    protocol = config.protocol()
    if cancelled and cancelled():
        raise Cancelled()
    if protocol["budget"] == 0:
        protocol["execution"] = "zero-budget identity; attack computation skipped"
        if callback:
            callback(1, 1, [])
        return images.detach().clone(), {
            "protocol": protocol,
            "trace": [],
            "dependencies": package_versions(),
        }
    classifier = NormalizedClassifier(model, spec, cancelled).eval()
    trace = []
    if config.algorithm in ("fgsm", "pgd", "pgd20", "pgd100", "cw_linf"):
        adversarial, trace = projected_attack(
            classifier, images, labels, config, callback
        )
    elif config.algorithm.startswith("deepfool"):
        try:
            import foolbox as fb
        except ImportError as exc:
            raise RuntimeError(
                "缺少 Foolbox，请安装 requirements-attacks.txt。"
            ) from exc
        if callback:
            callback(0, 1, [])
        attack_class = (
            fb.attacks.LinfDeepFoolAttack
            if protocol["norm"] == "Linf"
            else fb.attacks.L2DeepFoolAttack
        )
        attack = attack_class(
            steps=config.deepfool_steps,
            candidates=config.deepfool_candidates,
            overshoot=config.deepfool_overshoot,
        )
        wrapped = fb.PyTorchModel(classifier, bounds=(0, 1), device=images.device)
        with fixed_random_state(config.seed):
            _, adversarial, _ = attack(
                wrapped, images, labels, epsilons=protocol["budget"]
            )
    else:
        try:
            from autoattack import AutoAttack
        except ImportError as exc:
            raise RuntimeError(
                "缺少官方 AutoAttack，请安装 requirements-attacks.txt；不会退化成 PGD。"
            ) from exc
        if callback:
            callback(0, 1, [])
        if not package_versions()["autoattack_commit_verified"]:
            raise RuntimeError(
                "AA 来源不是已验证的固定官方提交，请重新安装 requirements-attacks.txt。"
            )
        with fixed_random_state(config.seed):
            attack = AutoAttack(
                classifier,
                norm="Linf",
                eps=protocol["budget"],
                seed=config.seed,
                version="standard",
                device=str(images.device),
                verbose=bool(log_path),
                log_path=str(log_path) if log_path else None,
            )
            if attack.attacks_to_run != protocol["components"]:
                raise RuntimeError("AA standard 的组件与固定协议不同，已停止评测。")
            protocol["component_settings"] = {
                "apgd_ce": {
                    "iterations": attack.apgd.n_iter,
                    "restarts": attack.apgd.n_restarts,
                },
                "apgd_targeted": {
                    "iterations": attack.apgd_targeted.n_iter,
                    "target_classes": attack.apgd_targeted.n_target_classes,
                },
                "fab_targeted": {
                    "iterations": attack.fab.n_iter,
                    "target_classes": attack.fab.n_target_classes,
                },
                "square": {
                    "queries": attack.square.n_queries,
                    "restarts": attack.square.n_restarts,
                },
            }
            adversarial = attack.run_standard_evaluation(images, labels, bs=len(images))
    if cancelled and cancelled():
        raise Cancelled()
    adversarial = adversarial.detach()
    if adversarial.shape != images.shape or not torch.isfinite(adversarial).all():
        raise ValueError("攻击输出形状或数值无效。")
    if float(adversarial.min()) < -1e-6 or float(adversarial.max()) > 1 + 1e-6:
        raise ValueError("攻击输出超出 [0,1] 像素范围。")
    delta = (adversarial - images).flatten(1)
    distances = (
        delta.abs().max(1).values
        if protocol["norm"] == "Linf"
        else delta.norm(p=2, dim=1)
    )
    if bool((distances > protocol["budget"] + 1e-5).any()):
        raise ValueError("攻击输出违反扰动预算，结果不会记入评测。")
    if callback and not trace:
        callback(1, 1, [])
    return adversarial, {
        "protocol": protocol,
        "trace": trace,
        "dependencies": package_versions(),
    }
