"""模型适配。输入为 [0, 1] 图像，归一化在模型内完成。"""
import torch
from ._core.engine import ARCHITECTURES


class Classifier(torch.nn.Module):
    def __init__(self, model, mean=(0, 0, 0), std=(1, 1, 1)):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).float().view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).float().view(1, -1, 1, 1))

    def forward(self, images):
        return self.model((images - self.mean) / self.std)


def create_model(architecture="cifar_resnet18", num_classes=10):
    return ARCHITECTURES[architecture](weights=None, num_classes=num_classes)
