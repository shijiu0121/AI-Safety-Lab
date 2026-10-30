"""攻击名称、有效参数与实验协议。网页和命令行都使用这一份定义。"""

from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

AUTOATTACK_COMMIT = "a39220048b3c9f2cca9a4d3a54604793c68eca7e"
CATALOG = {
    "fgsm": {"name": "FGSM", "norm": "Linf", "steps": 1, "backend": "PyTorch"},
    "pgd20": {"name": "PGD-20", "norm": "Linf", "steps": 20, "backend": "PyTorch"},
    "pgd100": {"name": "PGD-100", "norm": "Linf", "steps": 100, "backend": "PyTorch"},
    "pgd": {
        "name": "PGD（自定义步数）",
        "norm": "Linf",
        "steps": None,
        "backend": "PyTorch",
    },
    "cw_linf": {
        "name": "CW-L∞ · margin-PGD",
        "norm": "Linf",
        "steps": None,
        "backend": "PyTorch",
    },
    "deepfool_linf": {
        "name": "DeepFool-L∞",
        "norm": "Linf",
        "steps": None,
        "backend": "Foolbox",
    },
    "deepfool_l2": {
        "name": "DeepFool-L2",
        "norm": "L2",
        "steps": None,
        "backend": "Foolbox",
    },
    "autoattack": {
        "name": "AutoAttack · standard",
        "norm": "Linf",
        "steps": None,
        "backend": "official AutoAttack",
    },
}
AttackName = Literal[
    "fgsm",
    "pgd20",
    "pgd100",
    "pgd",
    "cw_linf",
    "deepfool_linf",
    "deepfool_l2",
    "autoattack",
]


class AttackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    algorithm: AttackName = "pgd20"
    epsilon_255: float = Field(default=8, ge=0, le=32)
    epsilon_l2: float = Field(default=0.5, ge=0, le=20)
    steps: int = Field(default=20, ge=1, le=500)
    step_size_255: float = Field(default=1, gt=0, le=32)
    restarts: int = Field(default=1, ge=1, le=20)
    cw_steps: int = Field(default=100, ge=1, le=500)
    cw_confidence: float = Field(default=0, ge=0, le=100)
    deepfool_steps: int = Field(default=50, ge=1, le=500)
    deepfool_candidates: int = Field(default=10, ge=2, le=1000)
    deepfool_overshoot: float = Field(default=0.02, ge=0, le=1)
    seed: int = Field(default=0, ge=0, le=2147483647)

    def protocol(self):
        descriptor = CATALOG[self.algorithm]
        budget = (
            self.epsilon_l2 if descriptor["norm"] == "L2" else self.epsilon_255 / 255
        )
        steps = descriptor["steps"]
        if self.algorithm == "pgd":
            steps = self.steps
        elif self.algorithm == "cw_linf":
            steps = self.cw_steps
        elif self.algorithm.startswith("deepfool"):
            steps = self.deepfool_steps
        return {
            **self.model_dump(),
            **descriptor,
            "iterations": steps,
            "budget": budget,
            "input_space": "RGB [0,1] before normalization",
            "random_start": self.algorithm in ("pgd", "pgd20", "pgd100", "cw_linf"),
            "step_size": (
                self.step_size_255 / 255
                if self.algorithm in ("pgd", "pgd20", "pgd100", "cw_linf")
                else None
            ),
            "actual_restarts": (
                self.restarts
                if self.algorithm in ("pgd", "pgd20", "pgd100", "cw_linf")
                else None
            ),
            "autoattack_version": (
                "standard" if self.algorithm == "autoattack" else None
            ),
            "autoattack_commit": (
                AUTOATTACK_COMMIT if self.algorithm == "autoattack" else None
            ),
            "components": (
                ["apgd-ce", "apgd-t", "fab-t", "square"]
                if self.algorithm == "autoattack"
                else None
            ),
            "paper_alignment": (
                "CW 损失公式、置信参数、步长与重启仍待原实验代码核对"
                if self.algorithm == "cw_linf"
                else None
            ),
        }


def check_attack_compatibility(config, number_of_classes):
    if config.algorithm == "autoattack" and number_of_classes < 10:
        raise ValueError(
            "官方 AA standard 预设使用 9 个目标类别，当前要求模型至少 10 类；不会悄悄删减组件。"
        )
