"""在服务器上对 CIFAR 官方测试集执行统一评测；结果来自真实推理。"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .engine import attack_tensor, forward, load_model, sha256
from .train import DATASETS


def main():
    parser = argparse.ArgumentParser(description="Aegis Lab GPU 批量评测")
    parser.add_argument(
        "--model", required=True, help="训练导出的 model_registration.json"
    )
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--data-dir", default="./data/datasets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--epsilon-255", type=float, default=8)
    parser.add_argument("--step-size-255", type=float, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=0, help="0 表示完整测试集；正数使用固定随机子集"
    )
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or args.steps < 1
        or args.limit < 0
        or not 0 <= args.epsilon_255 <= 32
        or args.step_size_255 <= 0
    ):
        parser.error("评测参数超出范围。")
    spec = json.loads(Path(args.model).read_text(encoding="utf-8"))
    if (
        spec["num_classes"] != DATASETS[args.dataset]["classes"]
        or spec["input_size"] != 32
    ):
        raise ValueError("模型类别数或输入尺寸与数据集不匹配。")
    dataset = DATASETS[args.dataset]["class"](
        args.data_dir, train=False, transform=transforms.ToTensor(), download=False
    )
    if spec.get("categories") != dataset.classes:
        raise ValueError("模型导出的类别顺序与数据集不一致。")
    spec.setdefault("id", "command_line_model")
    model, device = load_model(spec)
    generator = torch.Generator().manual_seed(args.seed)
    indices = (
        torch.randperm(len(dataset), generator=generator).tolist()
        if args.limit
        else list(range(len(dataset)))
    )
    if args.limit:
        indices = indices[: args.limit]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("请使用空的输出目录，避免覆盖已有评测结果。")
    (output_dir / "sample_indices.json").write_text(
        json.dumps(indices), encoding="utf-8"
    )
    clean_correct, robust_correct, successful_attacks = 0, 0, 0
    rows = []
    offset = 0
    for batch_index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            clean_predictions = forward(model, images, spec).argmax(1)
        adversarial, _ = attack_tensor(
            model,
            images,
            labels,
            spec,
            "pgd",
            args.epsilon_255 / 255,
            args.steps,
            args.step_size_255 / 255,
            args.seed + batch_index,
            record_trace=False,
        )
        with torch.no_grad():
            adversarial_predictions = forward(model, adversarial, spec).argmax(1)
        clean_ok = clean_predictions.eq(labels)
        robust_ok = clean_ok & adversarial_predictions.eq(labels)
        attack_ok = clean_ok & adversarial_predictions.ne(labels)
        clean_correct += int(clean_ok.sum())
        robust_correct += int(robust_ok.sum())
        successful_attacks += int(attack_ok.sum())
        for local_index in range(len(labels)):
            rows.append(
                {
                    "sample_index": indices[offset + local_index],
                    "label": int(labels[local_index]),
                    "clean_prediction": int(clean_predictions[local_index]),
                    "adversarial_prediction": int(adversarial_predictions[local_index]),
                    "linf": float(
                        (adversarial[local_index] - images[local_index]).abs().max()
                    ),
                }
            )
        offset += len(labels)
        print(f"已评测 {offset}/{len(indices)}", flush=True)
    with open(output_dir / "samples.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "dataset": args.dataset,
        "split": "official_test",
        "sample_count": len(indices),
        "clean_accuracy": clean_correct / len(indices),
        "pgd_robust_accuracy": robust_correct / len(indices),
        "attack_success_rate_on_clean_correct": (
            successful_attacks / clean_correct if clean_correct else None
        ),
        "clean_correct_count": clean_correct,
        "weights_sha256": sha256(spec["weights_path"]),
        "model": spec,
        "protocol": vars(args),
        "device": str(device),
        "torch": torch.__version__,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
