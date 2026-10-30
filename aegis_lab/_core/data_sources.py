"""只读数据集适配：官方 CIFAR 测试集和用户的 ImageFolder。不会自动下载。"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from torchvision import datasets, transforms


class DatasetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    kind: Literal["cifar10", "cifar100", "imagefolder"]
    root: str = Field(min_length=1, max_length=1500)
    split: Literal["test", "validation"] = "test"


def load_dataset(spec, image_size):
    root = Path(spec["root"]).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("数据目录不存在：请填写运行后端的机器上的目录。")
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    if spec["kind"] == "imagefolder":
        data = datasets.ImageFolder(str(root), transform=transform)
    else:
        if spec["split"] != "test":
            raise ValueError(
                "CIFAR 内置评测入口只使用官方 test；验证集请显式登记为独立 ImageFolder。"
            )
        dataset_class = (
            datasets.CIFAR10 if spec["kind"] == "cifar10" else datasets.CIFAR100
        )
        data = dataset_class(
            str(root), train=False, transform=transform, download=False
        )
    if len(data) < 1:
        raise ValueError("数据集为空。")
    return data


def inspect_dataset(spec):
    data = load_dataset(spec, 32)
    root = Path(spec["root"]).expanduser().resolve()
    if spec["kind"] == "imagefolder":
        files = [
            (
                str(Path(name).relative_to(root)),
                int(label),
                Path(name).stat().st_size,
                Path(name).stat().st_mtime_ns,
            )
            for name, label in data.samples
        ]
    else:
        files = [
            (str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        ]
    fingerprint = hashlib.sha256(
        json.dumps(files, ensure_ascii=False).encode()
    ).hexdigest()
    counts = [0] * len(data.classes)
    for label in data.targets:
        counts[int(label)] += 1
    return {
        "sample_count": len(data),
        "classes": list(data.classes),
        "class_counts": counts,
        "source_fingerprint": fingerprint,
        "fingerprint_type": "relative paths, labels, file sizes and mtimes",
        "note": "目录指纹是元数据摘要；运行时还会保存每批实际输入张量的 SHA256。",
    }


def check_dataset_model(dataset_spec, model_spec):
    inspection = dataset_spec.get("inspection")
    if not inspection:
        raise ValueError("请先检查数据集。")
    if model_spec["num_classes"] != len(inspection["classes"]):
        raise ValueError(f"模型 {model_spec['name']} 的类别数量与数据集不一致。")
    if (
        model_spec.get("categories")
        and model_spec["categories"] != inspection["classes"]
    ):
        raise ValueError(f"模型 {model_spec['name']} 的类别名称或顺序与数据集不一致。")
