"""训练数据与独立验证划分。绝不加载官方测试集来选择检查点。"""

import hashlib
import json
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATASETS = {
    "cifar10": dict(
        classes=10, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616]
    ),
    "cifar100": dict(
        classes=100, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
    ),
}

DATASETS["cifar10"]["class"] = datasets.CIFAR10
DATASETS["cifar100"]["class"] = datasets.CIFAR100


def folder_fingerprint(data):
    return hashlib.sha256(
        json.dumps(
            [
                (
                    str(Path(name).resolve()),
                    int(label),
                    Path(name).stat().st_size,
                    Path(name).stat().st_mtime_ns,
                )
                for name, label in data.samples
            ]
        ).encode()
    ).hexdigest()


def prepare_data(config, output_dir):
    size = config["input_size"]
    training_steps = [transforms.Resize((size, size))]
    if config.get("augmentation", "standard") == "randaugment":
        training_steps.append(transforms.RandAugment(num_ops=2, magnitude=8))
    elif config.get("augmentation", "standard") == "autoaugment":
        training_steps.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
    training_steps.extend([transforms.RandomCrop(size, padding=max(1, size // 8)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    training_transform = transforms.Compose(training_steps)
    validation_transform = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor()]
    )
    data_dir = str(Path(config["data_dir"]).resolve())
    if config["dataset"] == "imagefolder":
        training_data = datasets.ImageFolder(data_dir, transform=training_transform)
        validation_data = datasets.ImageFolder(
            config["validation_dir"] or data_dir, transform=validation_transform
        )
        if training_data.classes != validation_data.classes:
            raise ValueError("训练与验证目录的类别名称/顺序不一致。")
        if config["validation_dir"]:
            training_paths = {
                str(Path(item[0]).resolve()) for item in training_data.samples
            }
            validation_paths = {
                str(Path(item[0]).resolve()) for item in validation_data.samples
            }
            if training_paths & validation_paths:
                raise ValueError("训练和验证目录含有相同文件路径。")
        source = dict(
            train=folder_fingerprint(training_data),
            validation=folder_fingerprint(validation_data),
        )
        mean, std = config["mean"], config["std"]
    else:
        dataset_class = (
            datasets.CIFAR10 if config["dataset"] == "cifar10" else datasets.CIFAR100
        )
        training_data = dataset_class(
            data_dir,
            train=True,
            transform=training_transform,
            download=config["download"],
        )
        validation_data = dataset_class(
            data_dir, train=True, transform=validation_transform, download=False
        )
        source = {
            "train_data_sha256": hashlib.sha256(
                training_data.data.tobytes()
            ).hexdigest()
        }
        information = DATASETS[config["dataset"]]
        mean, std = (
            config["mean"] or information["mean"],
            config["std"] or information["std"],
        )
    classes = training_data.classes
    if len(classes) < 2:
        raise ValueError("分类训练至少需要两个类别。")
    generator = np.random.default_rng(config["seed"])
    labels = np.asarray(training_data.targets)
    training_indices, validation_indices = [], []
    for class_index in range(len(classes)):
        indices = generator.permutation(np.flatnonzero(labels == class_index)).tolist()
        if config["validation_dir"]:
            remaining = indices
        else:
            if len(indices) < 2:
                raise ValueError("每个类别至少需要两张图片用于分离训练/验证。")
            count = max(1, int(len(indices) * config["validation_fraction"]))
            validation_indices.extend(indices[:count])
            remaining = indices[count:]
        retained = max(
            1,
            int(
                len(remaining)
                * config["imbalance_ratio"] ** (-class_index / (len(classes) - 1))
            ),
        )
        training_indices.extend(remaining[:retained])
    if config["validation_dir"]:
        validation_indices = list(range(len(validation_data)))

    def limited(indices, labels, limit):
        if not limit or limit >= len(indices):
            return indices
        groups = {
            category: [index for index in indices if labels[index] == category]
            for category in sorted(set(labels[indices]))
        }
        if limit < len(groups):
            raise ValueError("样本上限至少应覆盖每个类别一张，避免静默丢失类别。")
        chosen = [int(generator.choice(group)) for group in groups.values()]
        chosen_set = set(chosen)
        remaining = [index for index in indices if index not in chosen_set]
        chosen.extend(generator.permutation(remaining)[: limit - len(chosen)].tolist())
        return chosen

    training_indices = limited(
        training_indices, labels, config.get("max_train_samples", 0)
    )
    validation_indices = limited(
        validation_indices,
        np.asarray(validation_data.targets),
        config.get("max_validation_samples", 0),
    )
    manifest = dict(
        dataset=config["dataset"],
        seed=config["seed"],
        training_indices=training_indices,
        validation_indices=validation_indices,
        class_counts=np.bincount(
            labels[training_indices], minlength=len(classes)
        ).tolist(),
        classes=classes,
        sample_limits=dict(
            training=config.get("max_train_samples", 0),
            validation=config.get("max_validation_samples", 0),
        ),
        official_test_used=False,
        source=source,
        source_digest_note="ImageFolder hashes metadata; CIFAR hashes loaded pixels",
    )
    path = Path(output_dir) / "split.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("数据或划分发生变化，不能复用已有训练目录。")
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    options = dict(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    from .train import seed_worker

    options["worker_init_fn"] = seed_worker
    train_loader = DataLoader(
        Subset(training_data, training_indices), shuffle=True, **options
    )
    val_loader = DataLoader(
        Subset(validation_data, validation_indices), shuffle=False, **options
    )
    return train_loader, val_loader, dict(classes=classes, mean=mean, std=std, class_counts=manifest["class_counts"])
