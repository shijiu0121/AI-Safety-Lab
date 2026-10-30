"""GPU 训练入口。

阅读顺序：load_configuration → prepare_data → train_one_epoch → main。
本文件不依赖网页。服务器上可以先单独验证训练，再启动工作台。
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .engine import ARCHITECTURES, attack_tensor, forward, select_device

from .training_config import load_configuration
from .training_data import DATASETS, prepare_data
from . import engine


def stop_requested():
    value = os.environ.get("AEGIS_STOP_FILE")
    return bool(value and Path(value).exists())


def check_stop():
    if stop_requested():
        raise engine.Cancelled("停止请求已收到；可从最近完成的 epoch 恢复。")


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    soft = (
        torch.nn.functional.kl_div(
            (student_logits / temperature).log_softmax(1),
            (teacher_logits.detach() / temperature).softmax(1),
            reduction="batchmean",
        )
        * temperature**2
    )
    return (
        alpha * torch.nn.functional.cross_entropy(student_logits, labels)
        + (1 - alpha) * soft
    )


def balanced_softmax_loss(logits, labels, class_counts):
    """AT-BSL：按训练集类别样本数调整 logits。"""
    counts = torch.as_tensor(class_counts, device=logits.device, dtype=logits.dtype)
    return torch.nn.functional.cross_entropy(logits + counts.log(), labels)


def hierarchical_equalization_loss(logits, labels, alpha=0.1, beta=0.1, gamma=0.1):
    """TAET 原作者训练 notebook 的类别损失、方差和归一化损失。"""
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    classes = logits.shape[1]
    sums = torch.zeros(classes, device=logits.device, dtype=logits.dtype).scatter_add(0, labels, losses)
    counts = torch.bincount(labels, minlength=classes).to(logits.dtype)
    class_losses = sums / (counts + 1e-10)
    average = class_losses.mean()
    normalized = class_losses / (class_losses.sum() + 1e-10)
    return alpha * average + beta * (class_losses - average).square().mean() + gamma * normalized.square().sum()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    # 每个数据读取进程使用不同但可追溯的种子。
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def generate_training_attack(model, images, labels, model_config, config, step_seed):
    """生成攻击时固定 BatchNorm 统计，随后恢复训练模式；攻击内部使用 float32。"""
    was_training = model.training
    model.eval()
    if config["method"] in ("at_bsl", "taet"):
        random_generator = torch.Generator(device=images.device).manual_seed(step_seed)
        adversarial = images.detach() + 0.001 * torch.randn(images.shape, device=images.device, generator=random_generator)
        bound = config["epsilon_255"] / 255
        for _ in range(config["attack_steps"]):
            check_stop()
            adversarial = adversarial.detach().requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(forward(model, adversarial, model_config), labels)
            gradient = torch.autograd.grad(loss, adversarial)[0]
            adversarial = adversarial.detach() + config["step_size_255"] / 255 * gradient.sign()
            adversarial = torch.maximum(torch.minimum(adversarial, images + bound), images - bound).clamp(0, 1)
    elif config["method"] == "trades":
        with torch.no_grad():
            target_distribution = forward(model, images, model_config).softmax(dim=1)
        random_generator = torch.Generator(device=images.device).manual_seed(step_seed)
        noise = (
            torch.rand(images.shape, device=images.device, generator=random_generator)
            * 2
            - 1
        )
        adversarial = (images + noise * config["epsilon_255"] / 255).clamp(0, 1)
        for _ in range(config["attack_steps"]):
            check_stop()
            adversarial = adversarial.detach().requires_grad_(True)
            divergence = torch.nn.functional.kl_div(
                forward(model, adversarial, model_config).log_softmax(dim=1),
                target_distribution,
                reduction="batchmean",
            )
            gradient = torch.autograd.grad(divergence, adversarial)[0]
            adversarial = (
                adversarial.detach() + config["step_size_255"] / 255 * gradient.sign()
            )
            bound = config["epsilon_255"] / 255
            adversarial = torch.maximum(
                torch.minimum(adversarial, images + bound), images - bound
            ).clamp(0, 1)
    else:
        adversarial, _ = attack_tensor(
            model,
            images,
            labels,
            model_config,
            "pgd",
            config["epsilon_255"] / 255,
            config["attack_steps"],
            config["step_size_255"] / 255,
            step_seed,
            record_trace=False,
            cancelled=stop_requested,
        )
    model.train(was_training)
    return adversarial.detach()


def train_one_epoch(
    model, loader, optimizer, model_config, config, device, epoch, teacher=None, class_counts=None
):
    model.train()
    total_loss, sample_count = 0.0, 0
    for batch_index, (images, labels) in enumerate(loader):
        check_stop()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if config["method"] == "distill_teacher":
            loss = torch.nn.functional.cross_entropy(
                forward(model, images, model_config) / config["temperature"], labels
            )
        elif config["method"] == "distill":
            teacher_model, teacher_spec = teacher
            teacher_images = images
            if teacher_spec["input_size"] != images.shape[-1]:
                teacher_images = torch.nn.functional.interpolate(
                    images,
                    size=(teacher_spec["input_size"],) * 2,
                    mode="bilinear",
                    align_corners=False,
                )
            with torch.no_grad():
                targets = forward(teacher_model, teacher_images, teacher_spec)
            loss = distillation_loss(
                forward(model, images, model_config),
                targets,
                labels,
                config["temperature"],
                config["distill_alpha"],
            )
        elif config["method"] == "erm" or (config["method"] == "taet" and epoch < config["taet_warmup_epochs"]):
            loss = torch.nn.functional.cross_entropy(
                forward(model, images, model_config), labels
            )
        else:
            step_seed = config["seed"] + epoch * len(loader) + batch_index
            adversarial = generate_training_attack(
                model, images, labels, model_config, config, step_seed
            )
            adversarial_logits = forward(model, adversarial, model_config)
            if config["method"] == "pgd_at":
                loss = torch.nn.functional.cross_entropy(adversarial_logits, labels)
            elif config["method"] == "at_bsl":
                loss = balanced_softmax_loss(adversarial_logits, labels, class_counts)
            elif config["method"] == "taet":
                loss = hierarchical_equalization_loss(adversarial_logits, labels, config["taet_alpha"], config["taet_beta"], config["taet_gamma"])
            else:
                clean_logits = forward(model, images, model_config)
                natural_loss = torch.nn.functional.cross_entropy(clean_logits, labels)
                robust_loss = torch.nn.functional.kl_div(
                    adversarial_logits.log_softmax(dim=1),
                    clean_logits.softmax(dim=1),
                    reduction="batchmean",
                )
                loss = natural_loss + config["trades_beta"] * robust_loss
        if not torch.isfinite(loss):
            raise RuntimeError("训练损失不是有限数值，已停止，防止保存无效模型。")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        sample_count += len(labels)
        if batch_index % 50 == 0:
            print(
                f"epoch={epoch + 1} batch={batch_index}/{len(loader)} loss={float(loss.detach()):.4f}",
                flush=True,
            )
    return total_loss / sample_count


def validate(model, loader, model_config, config, device):
    model.eval()
    clean_correct, robust_correct, total = 0, 0, 0
    per_class_total = torch.zeros(model_config["num_classes"], dtype=torch.long)
    per_class_robust = torch.zeros_like(per_class_total)
    for batch_index, (images, labels) in enumerate(loader):
        check_stop()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            clean_ok = forward(model, images, model_config).argmax(1).eq(labels)
        attacked, _ = attack_tensor(
            model,
            images,
            labels,
            model_config,
            "pgd",
            config["epsilon_255"] / 255,
            config["validation_steps"],
            config["step_size_255"] / 255,
            config["seed"] + 10_000_000 + batch_index,
            record_trace=False,
            cancelled=stop_requested,
        )
        with torch.no_grad():
            robust_ok = clean_ok & forward(model, attacked, model_config).argmax(1).eq(
                labels
            )
        clean_correct += int(clean_ok.sum())
        robust_correct += int(robust_ok.sum())
        total += len(labels)
        for class_index in range(model_config["num_classes"]):
            selected = labels.eq(class_index)
            per_class_total[class_index] += int(selected.sum())
            per_class_robust[class_index] += int((selected & robust_ok).sum())
    return {
        "validation_clean_accuracy": clean_correct / total,
        "validation_pgd_accuracy": robust_correct / total,
        "per_class_pgd_accuracy": [
            (
                float(per_class_robust[i] / per_class_total[i])
                if per_class_total[i]
                else None
            )
            for i in range(len(per_class_total))
        ],
    }


def atomic_save(payload, path):
    """先写临时文件再替换，减少中断时损坏上一检查点的概率。"""
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Aegis Lab GPU 训练入口")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="继续训练的检查点路径")
    parser.add_argument(
        "--check-only", action="store_true", help="只检查配置与 CUDA，不下载或训练"
    )
    parser.add_argument(
        "--stop-file",
        default=None,
        help="存在此文件时在安全边界停止；保留上一 epoch 检查点",
    )
    args = parser.parse_args()
    if args.stop_file:
        os.environ["AEGIS_STOP_FILE"] = args.stop_file
    config = load_configuration(args.config)
    device = select_device()
    if device.type != "cuda":
        raise RuntimeError("训练入口要求 CUDA；本机 CPU 只用于独立单元测试。")
    print(
        f"GPU: {torch.cuda.get_device_name(device)}; PyTorch: {torch.__version__}",
        flush=True,
    )
    if args.check_only:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    seed_everything(config["seed"])
    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_config_path = output_dir / "config.json"
    if saved_config_path.exists():
        previous = json.loads(saved_config_path.read_text(encoding="utf-8"))
        if previous != config or not args.resume:
            raise ValueError(
                "已有训练目录仅允许以相同配置显式恢复；新实验请使用新目录。"
            )
    else:
        saved_config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    training_loader, validation_loader, information = prepare_data(config, output_dir)
    categories = information["classes"]
    model_config = {
        "architecture": config["architecture"],
        "num_classes": len(categories),
        "input_size": config["input_size"],
        "mean": information["mean"],
        "std": information["std"],
        "categories": categories,
    }
    model = ARCHITECTURES[config["architecture"]](
        weights=None, num_classes=len(categories)
    ).to(device)
    teacher = None
    teacher_hashes = None
    if config["method"] == "distill":
        from .model_schema import model_payload

        raw_teacher = json.loads(
            Path(config["teacher_model_file"]).read_text(encoding="utf-8")
        )
        teacher_spec = dict(model_payload(raw_teacher), id="training_teacher")
        if (
            teacher_spec["num_classes"] != len(categories)
            or teacher_spec.get("categories")
            and teacher_spec["categories"] != categories
        ):
            raise ValueError("教师与训练数据的类别契约不一致。")
        teacher = (engine.load_model(teacher_spec)[0], teacher_spec)
        teacher_hashes = dict(
            registration=engine.sha256(config["teacher_model_file"]),
            weights=engine.sha256(teacher_spec["weights_path"]),
            loader=(
                engine.sha256(teacher_spec["loader_path"])
                if teacher_spec["architecture"] == "custom"
                else None
            ),
        )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )
    start_epoch, best_accuracy = 0, -1.0
    history, best_state = [], None
    from .training_state import atomic_tensor_save, publish_outputs

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint.get("teacher_hashes") != teacher_hashes:
            raise ValueError("教师权重或加载文件改变，不能恢复蒸馏。")
        if checkpoint["config"] != config:
            raise ValueError("检查点配置与本次训练配置不一致。")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint["best_validation_pgd"]
        if not {"history", "best_state_dict", "model_config"} <= set(checkpoint):
            raise ValueError(
                "旧检查点缺少一致恢复所需的最佳模型和指标，请创建新训练任务。"
            )
        history = checkpoint["history"]
        best_state = {
            key: value.cpu().clone()
            for key, value in checkpoint["best_state_dict"].items()
        }
        publish_outputs(checkpoint, output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, config["epochs"]):
        # 每个 epoch 重新设定种子，使按 epoch 恢复时数据增强与抽样有明确来源。
        seed_everything(config["seed"] + epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(
            model,
            training_loader,
            optimizer,
            model_config,
            config,
            device,
            epoch,
            teacher,
            information["class_counts"],
        )
        metrics = validate(model, validation_loader, model_config, config, device)
        row = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "training_loss": loss,
            **metrics,
        }
        scheduler.step()
        improved = metrics["validation_pgd_accuracy"] > best_accuracy
        best_accuracy = max(best_accuracy, metrics["validation_pgd_accuracy"])
        if improved or best_state is None:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        history.append(row)
        checkpoint = dict(
            teacher_hashes=teacher_hashes,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            epoch=epoch,
            best_validation_pgd=best_accuracy,
            config=config,
            best_state_dict=best_state,
            model_config=model_config,
            history=history,
        )
        # 检查点先作为提交点，曲线与导出模型随后发布；恢复时可重新生成。
        atomic_tensor_save(checkpoint, output_dir / "last.pt")
        publish_outputs(checkpoint, output_dir)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    try:
        main()
    except engine.Cancelled as exc:
        print(str(exc), flush=True)
        raise SystemExit(130)
