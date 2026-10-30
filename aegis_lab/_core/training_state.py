"""一个 epoch 的检查点是恢复依据；导出文件和曲线可以从它重建。"""

import json
from pathlib import Path
import torch


def atomic_tensor_save(body, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(body, temporary)
    temporary.replace(path)


def atomic_text(body, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


def publish_outputs(checkpoint, output_dir):
    """检查点已提交后发布衍生产物。进程中断时，下次恢复重新发布。"""
    output_dir = Path(output_dir)
    model_config = checkpoint["model_config"]
    weights_path = output_dir / "best_weights.pt"
    canonical = {
        key: value.detach().cpu()
        for key, value in checkpoint["best_state_dict"].items()
    }
    same = False
    if weights_path.is_file():
        try:
            existing = torch.load(weights_path, map_location="cpu", weights_only=True)
            same = (
                existing.get("aegis_model_config") == model_config
                and set(existing["state_dict"]) == set(canonical)
                and all(
                    torch.equal(existing["state_dict"][key], value)
                    for key, value in canonical.items()
                )
            )
        except (OSError, RuntimeError, KeyError, ValueError):
            same = False
    if not same:
        atomic_tensor_save(
            dict(state_dict=canonical, aegis_model_config=model_config), weights_path
        )
    export = dict(
        model_config,
        name=f'{checkpoint["config"]["dataset"]} / {checkpoint["config"]["method"]}',
        weights_path=str((output_dir / "best_weights.pt").resolve()),
    )
    atomic_text(
        json.dumps(export, ensure_ascii=False, indent=2),
        output_dir / "model_registration.json",
    )
    atomic_text(
        "".join(json.dumps(row) + "\n" for row in checkpoint["history"]),
        output_dir / "metrics.jsonl",
    )
