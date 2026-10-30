"""仅测试用的手工线性网络和合成图片；不是训练模型或性能数据。"""

from pathlib import Path
import torch
from PIL import Image
from backend import engine
from backend.data_sources import inspect_dataset


def fixture_resources(root):
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)
    loader = root / "fixture_model.py"
    loader.write_text(
        "from torch import nn\ndef build_model():\n    return nn.Sequential(nn.Flatten(),nn.Linear(768,2))\n",
        encoding="utf-8",
    )
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(768, 2))
    with torch.no_grad():
        model[1].weight[0].fill_(1 / 768)
        model[1].weight[1].fill_(-1 / 768)
        model[1].bias.copy_(torch.tensor([-0.5, 0.5]))
    weights = root / "fixture.pt"
    torch.save(model.state_dict(), weights)
    for label, value in [("A", 130), ("B", 120)]:
        folder = root / "images" / label
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            Image.new("RGB", (16, 16), (value, value, value)).save(
                folder / f"{index}.png"
            )
    spec = dict(
        id="fixture_model",
        name="测试线性网络 · 非性能结果",
        architecture="custom",
        weights_path=str(weights),
        loader_path=str(loader),
        factory="build_model",
        num_classes=2,
        input_size=16,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        categories=["A", "B"],
        status="validated",
    )
    spec["validation"] = engine.validate_model(spec)
    dataset = dict(
        id="fixture_data",
        name="六张合成图 · 功能夹具",
        kind="imagefolder",
        root=str(root / "images"),
        split="test",
        status="validated",
    )
    dataset["inspection"] = inspect_dataset(dataset)
    return spec, dataset
