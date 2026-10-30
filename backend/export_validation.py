"""将已保存的训练验证索引导出为 ImageFolder，供独立阈值校准。"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import datasets
from .training_config import load_configuration
from .batch_engine import save_json


def export_validation(config_file, split_file, output_dir):
    config = load_configuration(config_file)
    split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    if (
        split.get("official_test_used") is not False
        or split["dataset"] != config["dataset"]
    ):
        raise ValueError("划分不是本工程的训练验证划分。")
    if config["dataset"] == "imagefolder":
        root = Path(config["validation_dir"] or config["data_dir"]).resolve()
        data = datasets.ImageFolder(str(root))
    else:
        root = Path(config["data_dir"]).resolve()
        dataset_class = (
            datasets.CIFAR10 if config["dataset"] == "cifar10" else datasets.CIFAR100
        )
        data = dataset_class(str(root), train=True, download=False)
    destination = Path(output_dir).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("导出目录不能位于源数据根目录内。")
    indices = split["validation_indices"]
    if len(set(indices)) != len(indices) or any(
        index < 0 or index >= len(data) for index in indices
    ):
        raise ValueError("验证索引无效。")
    if not config["validation_dir"] and set(indices) & set(split["training_indices"]):
        raise ValueError("训练与验证索引重叠。")
    if list(data.classes) != split["classes"]:
        raise ValueError("当前数据类别顺序与训练划分不同。")
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(Path(split_file).read_bytes()).hexdigest()
    manifest_path = destination / "export_manifest.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8"))["split_sha256"]
        != fingerprint
    ):
        raise ValueError("输出目录已属于另一个划分。")
    for index in indices:
        image, label = data[index]
        image = image.convert("RGB")
        target = destination / data.classes[label] / f"validation_{index:08d}.png"
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            with Image.open(target) as existing:
                if not np.array_equal(np.asarray(existing), np.asarray(image)):
                    raise ValueError("已导出图片内容不一致，请使用新目录。")
        else:
            temporary = target.with_suffix(".tmp")
            image.save(temporary, format="PNG")
            temporary.replace(target)
    save_json(
        manifest_path,
        dict(
            split_sha256=fingerprint,
            sample_count=len(indices),
            source="training validation split; not official test",
            source_indices=indices,
        ),
    )
    return dict(
        id="calibration-validation",
        name="训练中独立保留的验证集",
        kind="imagefolder",
        root=str(destination),
        split="validation",
    )


def main():
    parser = argparse.ArgumentParser(description="导出用于校准的独立验证集")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_validation(args.config, args.split, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
