"""模型接入契约；API、资源导入与教师模型共用。"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=100)
    architecture: Literal[
        "resnet18",
        "resnet34",
        "mobilenet_v2",
        "cifar_resnet18",
        "cifar_resnet34",
        "densenet121",
        "efficientnet_b0",
        "wide_resnet50_2",
        "custom",
    ]
    weights_path: str = Field(min_length=1, max_length=1500)
    loader_path: str = ""
    factory: str = "build_model"
    num_classes: int = Field(default=1000, ge=2, le=10000)
    input_size: int = Field(default=224, ge=16, le=1024)
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]
    categories: list[str] = []

    @model_validator(mode="after")
    def contract(self):
        if (
            len(self.mean) != 3
            or len(self.std) != 3
            or any(item <= 0 for item in self.std)
        ):
            raise ValueError("mean/std 必须各有三个有限值，std 为正。")
        if self.categories and len(self.categories) != self.num_classes:
            raise ValueError("类别名称数量必须等于输出类别数。")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("类别名称不能重复。")
        if self.architecture == "custom" and not self.loader_path:
            raise ValueError("自定义模型需要加载文件路径。")
        return self


def model_payload(body):
    """仅移除导出记录的系统元数据，未知业务字段仍明确报错。"""
    metadata = {
        "id",
        "status",
        "created_at",
        "updated_at",
        "validation",
        "error",
        "archived",
        "weights_sha256",
        "loader_sha256",
        "training_run_id",
    }
    return ModelInput.model_validate(
        {key: value for key, value in body.items() if key not in metadata}
    ).model_dump()
