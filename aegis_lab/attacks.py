"""攻击接口与工作台使用同一套实现。"""
import torch
from ._core.attack_protocol import AttackConfig
from ._core.attacks import run_attack


def attack(model, images, labels, method="pgd20", return_details=False, **options):
    """返回对抗图像；epsilon_255、step_size_255 以像素级 255 为单位。"""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            classes = model(images[:1]).shape[1]
        channels = images.shape[1]
        spec = dict(num_classes=classes, mean=[0] * channels, std=[1] * channels)
        result, details = run_attack(model, images, labels, spec, AttackConfig(algorithm=method, **options))
        return (result, details) if return_details else result
    finally:
        for module, training in modes:
            module.training = training
