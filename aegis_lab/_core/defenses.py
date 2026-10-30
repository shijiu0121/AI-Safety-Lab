"""输入处理、模型集成与输出检测。"""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import engine


class TransformConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Literal["bit_depth", "median", "gaussian", "resize_pad", "noise"]
    bits: int = Field(default=5, ge=1, le=8)
    kernel: int = Field(default=3, ge=1, le=9)
    sigma: float = Field(default=0.5, gt=0, le=5)
    scale: float = Field(default=0.9, ge=0.5, le=1)
    noise_std: float = Field(default=0.01, ge=0, le=0.2)

    @model_validator(mode="after")
    def odd_kernel(self):
        if self.kernel % 2 != 1:
            raise ValueError("滤波核必须为奇数。")
        return self


class DetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    method: Literal["none", "confidence", "entropy", "consistency"] = "none"
    threshold: float = Field(default=0.5, ge=0, le=1)
    min_input_std: float = Field(default=0, ge=0, le=0.5)
    max_saturation: float = Field(default=1, ge=0, le=1)


class DefenseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=100)
    model_ids: list[str] = Field(min_length=1, max_length=8)
    aggregation: Literal["soft_vote", "logit_mean", "hard_vote", "random_member"] = (
        "soft_vote"
    )
    transforms: list[TransformConfig] = Field(default=[], max_length=8)
    detector: DetectorConfig = DetectorConfig()
    prediction_samples: int = Field(default=4, ge=1, le=32)
    eot_samples: int = Field(default=4, ge=1, le=32)
    seed: int = Field(default=0, ge=0, le=2000000000)

    @model_validator(mode="after")
    def distinct_models(self):
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("集成成员不能重复。")
        return self


def transform_images(images, transforms, generator=None, bpda=False):
    result = images
    for options in transforms:
        if not isinstance(options, TransformConfig):
            options = TransformConfig.model_validate(options)
        before = result
        if options.kind == "bit_depth":
            levels = 2**options.bits - 1
            result = (result * levels).round() / levels
        elif options.kind == "median":
            size = options.kernel
            padded = F.pad(result, (size // 2,) * 4, mode="reflect")
            windows = padded.unfold(2, size, 1).unfold(3, size, 1)
            result = windows.contiguous().flatten(-2).median(-1).values
        elif options.kind == "gaussian":
            axis = (
                torch.arange(options.kernel, device=result.device, dtype=result.dtype)
                - options.kernel // 2
            )
            kernel = torch.exp(-axis.square() / (2 * options.sigma**2))
            kernel = torch.outer(kernel, kernel)
            kernel = (kernel / kernel.sum()).expand(result.shape[1], 1, -1, -1)
            result = F.conv2d(
                F.pad(result, (options.kernel // 2,) * 4, mode="reflect"),
                kernel,
                groups=result.shape[1],
            )
        elif options.kind == "resize_pad":
            height, width = result.shape[-2:]
            smaller = (
                max(1, round(height * options.scale)),
                max(1, round(width * options.scale)),
            )
            result = F.interpolate(
                result, size=smaller, mode="bilinear", align_corners=False
            )
            top = (height - smaller[0]) // 2
            left = (width - smaller[1]) // 2
            result = F.pad(
                result,
                (left, width - smaller[1] - left, top, height - smaller[0] - top),
                value=0.5,
            )
        else:
            noise = torch.randn(
                result.shape,
                device=result.device,
                dtype=result.dtype,
                generator=generator,
            )
            result = result + noise * options.noise_std
        result = result.clamp(0, 1)
        # 前向严格使用真实变换；不可微操作的反向明确采用恒等近似。
        if bpda and options.kind in ("bit_depth", "median"):
            result = before + (result - before).detach()
    return result


def validate_members(config, specs):
    if [item["id"] for item in specs] != config.model_ids:
        raise ValueError("成员顺序必须与防御配置一致。")
    first = specs[0]
    for spec in specs:
        if spec.get("status") != "validated":
            raise ValueError("防御成员必须先通过模型验证。")
        for key in ("num_classes", "input_size", "categories"):
            if spec.get(key, []) != first.get(key, []):
                raise ValueError("当前集成要求输入大小、类别数和类别顺序相同。")
        if engine.sha256(spec["weights_path"]) != spec["validation"]["sha256"]:
            raise ValueError("成员权重已改变，请重新验证模型。")
        if spec["architecture"] == "custom" and engine.sha256(
            spec["loader_path"]
        ) != spec["validation"].get("loader_sha256"):
            raise ValueError("自定义加载文件已改变，请重新验证。")


class DefenseRuntime:
    def __init__(self, config, specs):
        self.config = (
            config
            if isinstance(config, DefenseConfig)
            else DefenseConfig.model_validate(config)
        )
        validate_members(self.config, specs)
        self.specs = specs
        loaded = [engine.load_model(spec) for spec in specs]
        self.models = [item[0] for item in loaded]
        self.device = loaded[0][1]
        self.stochastic = self.config.aggregation == "random_member" or any(
            item.kind == "noise" and item.noise_std > 0
            for item in self.config.transforms
        )
        self.bpda = any(
            item.kind in ("bit_depth", "median") for item in self.config.transforms
        )
        self.attack_calls = 0
        self.attack_seed = self.config.seed

    def logits(self, images, seed=0, surrogate=False):
        generator = torch.Generator(device=images.device).manual_seed(seed)
        count = (
            (self.config.eot_samples if surrogate else self.config.prediction_samples)
            if self.stochastic
            else 1
        )
        results = []
        for _ in range(count):
            transformed = transform_images(
                images, self.config.transforms, generator, bpda=surrogate
            )
            indices = list(range(len(self.models)))
            if self.config.aggregation == "random_member":
                indices = [
                    int(
                        torch.randint(
                            len(indices), (), generator=generator, device=images.device
                        )
                    )
                ]
            outputs = torch.stack(
                [
                    engine.forward(self.models[index], transformed, self.specs[index])
                    for index in indices
                ]
            )
            if self.config.aggregation == "logit_mean":
                results.append(outputs.mean(0).softmax(1))
            elif self.config.aggregation == "hard_vote" and not surrogate:
                votes = F.one_hot(outputs.argmax(2), outputs.shape[2]).to(outputs.dtype)
                results.append(votes.mean(0))
            else:
                # 硬投票的攻击使用软投票代理，但最终成功判据仍来自真实硬投票。
                results.append(outputs.softmax(2).mean(0))
        return torch.stack(results).mean(0).clamp_min(1e-12).log()

    def attack_logits(self, images):
        self.attack_calls += 1
        return self.logits(images, self.attack_seed + self.attack_calls, surrogate=True)

    def measure(self, images, seed=0):
        with torch.no_grad():
            logits = self.logits(images, seed)
            probabilities = logits.softmax(1)
            confidence, prediction = probabilities.max(1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
                1
            ) / math.log(probabilities.shape[1])
            detector = self.config.detector
            if detector.method == "confidence":
                scores = 1 - confidence
            elif detector.method == "entropy":
                scores = entropy
            elif detector.method == "consistency":
                reference = transform_images(images, [TransformConfig(kind="median")])
                other = self.logits(reference, seed).softmax(1)
                scores = 0.5 * (probabilities - other).abs().sum(1)
            else:
                scores = torch.zeros_like(confidence)
            scores = scores.clamp(0, 1)
            flat = images.flatten(1)
            input_std = flat.std(1, unbiased=False)
            saturation = ((flat <= 1 / 255) | (flat >= 254 / 255)).float().mean(1)
            anomaly = (input_std < detector.min_input_std) | (
                saturation > detector.max_saturation
            )
            detected = (
                (scores > detector.threshold)
                if detector.method != "none"
                else torch.zeros_like(anomaly)
            )
            return dict(
                logits=logits,
                prediction=prediction,
                confidence=confidence,
                entropy=entropy,
                score=scores,
                input_std=input_std,
                saturation=saturation,
                anomaly=anomaly,
                detected=detected,
                flagged=anomaly | detected,
            )

    def protocol(self):
        return dict(
            config=self.config.model_dump(),
            bpda_identity=self.bpda,
            eot_samples=self.config.eot_samples if self.stochastic else 1,
            prediction_samples=self.config.prediction_samples if self.stochastic else 1,
            hard_vote_surrogate=self.config.aggregation == "hard_vote",
            detector_attack_objective="classification loss",
        )


class AttackSurrogate(torch.nn.Module):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime

    def forward(self, images):
        return self.runtime.attack_logits(images)


def identity_spec(spec):
    return dict(spec, mean=[0, 0, 0], std=[1, 1, 1])
