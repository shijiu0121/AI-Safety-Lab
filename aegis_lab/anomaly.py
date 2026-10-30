"""输入统计、输出能量和滑动窗口分布偏移。"""
import torch
from .detection import Detector, device_of


class Energy(Detector):
    def __init__(self, model, temperature=1.0):
        super().__init__(model, temperature=temperature)

    def score(self, images):
        temperature = self.options["temperature"]
        return -temperature * torch.logsumexp(self.model(images) / temperature, dim=1)


class InputStatistics(Detector):
    @staticmethod
    def features(images):
        flat = images.flatten(2)
        mean = flat.mean(2)
        std = flat.std(2, unbiased=False)
        saturation = ((flat < 1 / 255) | (flat > 254 / 255)).float().mean(2)
        gradient = (images[:, :, 1:] - images[:, :, :-1]).abs().flatten(2).mean(2)
        return torch.cat([mean, std, saturation, gradient], 1)

    def fit(self, loader, **options):
        values = torch.cat([self.features(images).cpu() for images, _ in loader]).to(device_of(self.model))
        self.center = values.mean(0)
        self.scale = values.std(0, unbiased=False).clamp_min(1e-5)
        return self

    def score(self, images):
        return ((self.features(images) - self.center) / self.scale).abs().max(1).values


class OutputDrift:
    """输出概率分布的 Jensen-Shannon 距离；按固定窗口累积。"""
    def __init__(self, window_size=256):
        self.window_size = window_size
        self.pending = []

    def divergence(self, probabilities):
        current = probabilities.mean(0).clamp_min(1e-12)
        reference = self.reference.to(current).clamp_min(1e-12)
        middle = (current + reference) / 2
        return 0.5 * ((current * (current / middle).log()).sum() + (reference * (reference / middle).log()).sum())

    def fit(self, reference_probabilities):
        self.reference = reference_probabilities.detach().mean(0).cpu()
        return self

    def calibrate(self, probabilities, false_positive_rate=0.05):
        windows = probabilities.split(self.window_size)
        values = torch.stack([self.divergence(part) for part in windows if len(part) == self.window_size])
        self.threshold = float(torch.quantile(values, 1 - false_positive_rate, interpolation="higher"))
        return self

    def update(self, probabilities):
        self.pending.extend(probabilities.detach().cpu().unbind())
        events = []
        while len(self.pending) >= self.window_size:
            window = torch.stack(self.pending[:self.window_size])
            del self.pending[:self.window_size]
            score = float(self.divergence(window))
            events.append(dict(score=score, threshold=self.threshold, flagged=score > self.threshold, samples=self.window_size))
        return events

    def save(self, path):
        torch.save(dict(window_size=self.window_size, reference=self.reference, threshold=self.threshold), path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, weights_only=True)
        monitor = cls(state.pop("window_size"))
        monitor.__dict__.update(state)
        return monitor
