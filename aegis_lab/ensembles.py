"""多模型投票和随机化集成，输出可供攻击接口使用的 log probabilities。"""
import torch
from torch import nn
from torch.nn import functional as F
from .preprocessing import preprocess


class Ensemble(nn.Module):
    def __init__(self, models, method="soft_vote", transforms=(), samples=4, surrogate=False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method
        self.transforms = list(transforms)
        self.samples = samples
        self.surrogate = surrogate

    def forward(self, images):
        stochastic = self.method == "random_member" or any(t["kind"] == "noise" for t in self.transforms)
        predictions = []
        for _ in range(self.samples if stochastic else 1):
            inputs = preprocess(images, self.transforms, bpda=self.surrogate)
            members = [self.models[torch.randint(len(self.models), ()).item()]] if self.method == "random_member" else self.models
            logits = torch.stack([model(inputs) for model in members])
            if self.method == "logit_mean":
                probabilities = logits.mean(0).softmax(1)
            elif self.method == "hard_vote" and not self.surrogate:
                probabilities = F.one_hot(logits.argmax(2), logits.shape[2]).float().mean(0)
            else:
                probabilities = logits.softmax(2).mean(0)
            predictions.append(probabilities)
        return torch.stack(predictions).mean(0).clamp_min(1e-12).log()
