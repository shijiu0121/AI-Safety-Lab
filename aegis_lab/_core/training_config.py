"""可从网页导出的严格训练配置。"""

import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .engine import ARCHITECTURES


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    dataset: Literal["cifar10", "cifar100", "imagefolder"] = "cifar10"
    data_dir: str = "./data/datasets"
    validation_dir: str = ""
    output_dir: str = "./data/training/run"
    architecture: str = "cifar_resnet18"
    method: Literal["erm", "pgd_at", "trades", "at_bsl", "taet", "distill_teacher", "distill"] = "pgd_at"
    epochs: int = Field(default=100, ge=1, le=10000)
    batch_size: int = Field(default=128, ge=1, le=2048)
    input_size: int = Field(default=32, ge=16, le=512)
    mean: list[float] | None = None
    std: list[float] | None = None
    learning_rate: float = Field(default=0.1, gt=0, le=10)
    momentum: float = Field(default=0.9, ge=0, lt=1)
    weight_decay: float = Field(default=0.0005, ge=0, le=1)
    epsilon_255: float = Field(default=8, ge=0, le=32)
    attack_steps: int = Field(default=10, ge=1, le=500)
    step_size_255: float = Field(default=2, gt=0, le=32)
    validation_steps: int = Field(default=20, ge=1, le=500)
    validation_fraction: float = Field(default=0.1, gt=0, lt=0.5)
    num_workers: int = Field(default=4, ge=0, le=32)
    seed: int = Field(default=0, ge=0, le=2000000000)
    trades_beta: float = Field(default=6, gt=0, le=100)
    taet_warmup_epochs: int = Field(default=40, ge=0, le=10000)
    taet_alpha: float = Field(default=0.1, ge=0)
    taet_beta: float = Field(default=0.1, ge=0)
    taet_gamma: float = Field(default=0.1, ge=0)
    download: bool = False
    imbalance_ratio: float = Field(default=1, ge=1, le=10000)
    augmentation: Literal["standard", "randaugment", "autoaugment"] = "standard"
    temperature: float = Field(default=20, gt=0, le=1000)
    distill_alpha: float = Field(default=0.5, ge=0, le=1)
    teacher_model_file: str = ""

    max_train_samples: int = Field(default=0, ge=0, le=1000000)
    max_validation_samples: int = Field(default=0, ge=0, le=1000000)

    @model_validator(mode="after")
    def settings(self):
        if self.validation_dir and self.dataset != "imagefolder":
            raise ValueError("独立验证目录仅用于 ImageFolder；CIFAR 从官方训练集划分。")
        if self.architecture not in ARCHITECTURES:
            raise ValueError("不支持的模型结构。")
        if self.dataset == "imagefolder" and (self.mean is None or self.std is None):
            raise ValueError("ImageFolder 训练必须显式提供 mean/std。")
        for values in (self.mean, self.std):
            if values is not None and len(values) != 3:
                raise ValueError("mean/std 各需要三个数值。")
        if self.std is not None and any(item <= 0 for item in self.std):
            raise ValueError("std 必须为正。")
        if self.method == "distill" and not self.teacher_model_file:
            raise ValueError("学生蒸馏需要教师模型注册 JSON 路径。")
        return self


def load_configuration(path):
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(body) - set(TrainingConfig.model_fields)
    if unknown:
        raise ValueError(f"未知配置项：{sorted(unknown)}")
    return TrainingConfig.model_validate(body).model_dump()
