"""自定义接入示例：用你自己的网络定义替换这里，权重在工作台中单独指定。"""

from torch import nn


class SmallClassifier(nn.Module):
    def __init__(self, number_of_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, number_of_classes)

    def forward(self, images):
        features = self.features(images).flatten(1)
        return self.classifier(features)


def build_model():
    return SmallClassifier(number_of_classes=10)
