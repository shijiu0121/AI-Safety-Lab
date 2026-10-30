"""按模块名称读取中间特征，不要求用户改写模型的 forward。"""
import torch


def activations(model, images, layers):
    values, handles = {}, []
    modules = dict(model.named_modules())
    for name in dict.fromkeys(layers):
        def capture(module, inputs, output, key=name):
            values[key] = output.clone()
        handles.append(modules[name].register_forward_hook(capture))
    try:
        logits = model(images)
        return logits, [values[name] for name in layers]
    finally:
        for handle in handles:
            handle.remove()


def pooled(value):
    return value.flatten(2).mean(2) if value.ndim > 2 else value


def default_layers(architecture):
    if "resnet" in architecture:
        return ["model.layer2", "model.layer3"], "model.avgpool"
    if "densenet" in architecture:
        return ["model.features.denseblock2", "model.features.denseblock3"], "model.features.norm5"
    if architecture == "mobilenet_v2":
        return ["model.features.6", "model.features.13"], "model.features.18"
    raise ValueError("请在检测配置中指定 layers 和 target_layer。")
