"""可串联的输入变换；bpda=True 时为量化和中值滤波使用直通梯度。"""
from ._core.defenses import TransformConfig, transform_images


def preprocess(images, transforms, *, bpda=False, generator=None):
    return transform_images(images, transforms, generator=generator, bpda=bpda)
