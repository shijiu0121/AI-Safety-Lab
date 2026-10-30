"""现有 DataLoader 和 optimizer 可直接接入训练循环。"""
import torch
from ._core.training_config import TrainingConfig
from ._core.train import train_one_epoch, balanced_softmax_loss, hierarchical_equalization_loss, distillation_loss


def train_epoch(model, loader, optimizer, *, method="pgd_at", epoch=0, num_classes=10,
                class_counts=None, teacher=None, **options):
    """model/teacher 均接收 [0,1] 图像，可先使用 Classifier 包装归一化。"""
    config = TrainingConfig(method=method, teacher_model_file="in-memory" if teacher is not None else "", **options).model_dump()
    device = next(model.parameters()).device
    spec = dict(num_classes=num_classes, mean=[0, 0, 0], std=[1, 1, 1], input_size=config["input_size"])
    teacher_pair = (teacher, spec) if teacher is not None else None
    return train_one_epoch(model, loader, optimizer, spec, config, device, epoch, teacher_pair, class_counts)
