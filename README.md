<div align="center">

# Aegis Lab

**面向 PyTorch 图像分类的 AI 安全算法库与实验成果展示网站**

对抗攻击 · 鲁棒性训练 · 攻击检测 · 异常检测 · 统一评估

[快速开始](#快速开始) · [功能模块](#功能模块) · [实验复现](#实验复现) · [成果网站](#成果网站) · [API 文档](docs/API.md)

</div>

---

## 项目介绍

Aegis Lab 将对抗样本生成、鲁棒训练、检测与防御评估组织成一套可直接调用的 Python 接口。用户可以接入自己的 PyTorch 模型，在现有代码中使用单个算法，也可以通过统一实验配置完成训练、评估与成果导出。

项目由三个部分组成：

| 组成 | 作用 | 入口 |
| --- | --- | --- |
| **Python 算法库** | 在 Python 项目中调用攻击、训练、检测、预处理、集成与评估方法 | `aegis_lab` |
| **公开成果网站** | 介绍方法，比较实验结果，查看模型卡，下载配置、权重和报告 | `src/PublicSite.vue` |
| **可选实验工作台** | 通过图形界面接入模型、登记数据、配置任务和查看运行记录 | `#/workbench/overview` |

算法库与工作台复用核心实现。公开网站读取导出的静态结果，训练和评估在使用者自己的电脑或 GPU 服务器上执行。

### 项目特点

- **面向已有模型**：接收 `torch.nn.Module`，通过 `Classifier` 统一处理图像归一化。
- **统一检测接口**：检测方法共用 `fit → calibrate → score / predict → save` 流程。
- **覆盖常规与长尾训练**：集成 ERM、PGD-AT、TRADES、AT-BSL、TAET 和蒸馏训练。
- **完整实验产物**：保存训练配置、数据划分、检查点、指标、代表样本和依赖版本。
- **实验与展示衔接**：将结果导出为 JSON、CSV、HTML 和图像，网站直接读取并展示。
- **按需使用**：可以只安装算法库，也可以同时运行网站与工作台。

## 功能模块

### 对抗样本生成

| 方法 | 调用名称 | 实现方式 |
| --- | --- | --- |
| FGSM | `fgsm` | 单步梯度符号攻击 |
| PGD | `pgd` | 自定义迭代次数的投影梯度攻击 |
| PGD-20 / PGD-100 | `pgd20` / `pgd100` | 固定 20 / 100 次迭代 |
| CW-L∞ | `cw_linf` | CW margin 损失与 L∞ 投影迭代 |
| DeepFool | `deepfool_linf` / `deepfool_l2` | 调用 Foolbox 的 L∞ / L2 实现 |
| AutoAttack | `autoattack` | 官方 standard 组合：APGD-CE、APGD-T、FAB-T、Square |

攻击接口支持扰动预算、步长、随机种子和重启次数。`return_details=True` 时同时返回攻击协议、迭代记录和依赖版本。

### 攻击检测

| 方法 | 检测依据 | 拟合内容 |
| --- | --- | --- |
| Confidence | 最大预测置信度降低 | 校准阈值 |
| Entropy | 输出概率分布的归一化熵 | 校准阈值 |
| Consistency | 中值滤波前后的预测差异 | 校准阈值 |
| Feature Squeezing | 位深压缩、中值平滑前后的概率 L1 距离 | 校准阈值 |
| Mahalanobis | 类条件特征高斯分布的马氏距离 | 类别均值、共享协方差、阈值 |
| Layer Regression | 中间层预测深层特征的回归误差 | 两隐藏层 MLP、阈值 |

所有检测器都以“分数越大，越异常”为约定。阈值使用独立干净数据校准，检测器参数与阈值可保存并重新加载。

Mahalanobis 支持指定特征层、输入梯度预处理和逻辑回归融合。Layer Regression 支持指定中间层、目标层和隐藏层宽度。具体公式与选用配置见 [API 文档](docs/API.md)。

### 鲁棒性训练与蒸馏

| 方法 | 配置名称 | 核心内容 |
| --- | --- | --- |
| ERM | `erm` | 干净样本交叉熵基准 |
| PGD-AT | `pgd_at` | PGD 对抗样本训练 |
| TRADES | `trades` | 分类损失与预测分布 KL 鲁棒项 |
| AT-BSL | `at_bsl` | 对抗训练与类别频次修正的 Balanced Softmax |
| TAET | `taet` | 干净预训练与类别损失均衡的两阶段对抗训练 |
| 高温教师 | `distill_teacher` | 温度缩放的教师训练 |
| 学生蒸馏 | `distill` | 硬标签损失与教师软标签 KL |

普通知识蒸馏可以接入常规教师；防御蒸馏使用高温教师与学生训练流程。训练支持保存检查点、恢复 epoch、导出模型和记录验证曲线。

### 输入预处理与模型集成

- **输入预处理**：位深压缩、中值滤波、高斯滤波、缩放填充、随机噪声，支持按列表顺序组合。
- **模型集成**：软投票、logits 平均、硬投票、随机成员与多次采样。
- **自适应评估**：量化和中值滤波支持 BPDA，随机策略支持采样平均，硬投票使用软投票梯度代理生成攻击，再用实际投票评估。

### 异常检测

| 方法 | 检测对象 |
| --- | --- |
| InputStatistics | 通道均值、标准差、饱和率和相邻像素差异 |
| Energy | 基于 logits 能量分数的分布外输入 |
| Mahalanobis | 偏离类别特征分布的输入 |
| OutputDrift | 固定窗口内输出概率分布的 Jensen-Shannon 偏移 |

`OutputDrift` 支持跨批次累积窗口，输出分数、阈值、窗口样本数与告警状态。原工作台同时保留本机实时推理与事件记录功能。

### 安全评估与成果导出

评估指标包括：

- 干净准确率、鲁棒准确率、均衡准确率与逐类准确率；
- 攻击成功率、扰动大小、评估耗时；
- 检测 AUROC、TPR、FPR、FPR@95%TPR；
- 自适应攻击下的检测规避成功率；
- 输出分布偏移的窗口事件。

鲁棒准确率统计原始图像与对抗图像均分类正确的样本；攻击成功率以原图分类正确的样本为分母。完整指标约定见 [API 文档](docs/API.md)。

## 安装

### 环境

| 组件 | 要求 / 用途 |
| --- | --- |
| Python | 3.10 或以上 |
| PyTorch / torchvision | `torch>=2.6`、`torchvision>=0.21` |
| CUDA GPU | 命令行训练与正式 GPU 实验 |
| Node.js / npm | Vue / Vite 网站开发与构建；使用兼容 Vite 7 的 Node.js |
| FastAPI / Uvicorn | 可选工作台后端 |

克隆或下载仓库后，在工程根目录执行以下命令。GPU 服务器先安装与本机 CUDA 环境匹配的 PyTorch 和 torchvision。

### 从源码安装

```bash
# 基础算法库
python -m pip install -e .

# 包含 Foolbox 与官方 AutoAttack
python -m pip install -e ".[attacks]"

# 同时安装工作台依赖
python -m pip install -e ".[attacks,workbench]"

# 查看命令
python -m aegis_lab --help
```

安装后可使用 `aegis` 命令，也可以用 `python -m aegis_lab` 执行同样的操作。

### 构建安装包

```bash
python -m pip install build
python -m build --outdir packages
```

生成 wheel 与源码包后，可通过本地文件安装：

```bash
python -m pip install ./packages/aegis_lab-0.4.0-py3-none-any.whl
```

wheel 包含算法库和内置实验计划；网站与工作台源码位于仓库中。

## 快速开始

### 1. 接入模型并生成对抗样本

下面以项目训练导出的 CIFAR-10 权重为例。`test_loader` 为用户已有的 PyTorch DataLoader。

```python
import torch
from aegis_lab import Classifier, create_model, attack

net = create_model("cifar_resnet18", num_classes=10)
checkpoint = torch.load("best_weights.pt", map_location="cpu", weights_only=True)
net.load_state_dict(checkpoint["state_dict"])

model = Classifier(
    net,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.247, 0.2435, 0.2616),
).cuda().eval()

images, labels = next(iter(test_loader))
images, labels = images.cuda(), labels.cuda()

adversarial = attack(
    model, images, labels,
    method="pgd20",
    epsilon_255=8,
    step_size_255=2,
)
predictions = model(adversarial).argmax(dim=1)
```

输入图像为 `[N, 3, H, W]`、float32、像素范围 `[0,1]`；标签为整数类别编号。`Classifier` 在前向过程中归一化输入，`net` 输出 `[N,C]` logits。使用自定义模型时，将 `net` 替换为自己的 `torch.nn.Module`。

### 2. 校准并保存检测器

```python
from aegis_lab.detection import Mahalanobis, load_detector

# 层名来自 model.named_modules()
detector = Mahalanobis(model, layers=["model.avgpool"])
detector.fit(fit_loader)
detector.calibrate(calibration_loader, false_positive_rate=0.05)

scores = detector.score(adversarial)
flags = detector.predict(adversarial)

detector.save("detector.pt")
detector = load_detector("detector.pt", model)
```

`fit_loader` 提供拟合特征分布的数据，`calibration_loader` 提供独立干净校准数据。Feature Squeezing 等不需要拟合分布的方法可直接调用 `calibrate`。

### 3. 接入自己的训练循环

```python
from aegis_lab.training import train_epoch

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
for epoch in range(100):
    loss = train_epoch(
        model, train_loader, optimizer,
        method="at_bsl",
        epoch=epoch,
        num_classes=10,
        class_counts=class_counts,
        attack_steps=10,
        epsilon_255=8,
        step_size_255=2,
    )
```

`class_counts` 为实际训练划分的每类样本数。数据增强由 DataLoader 决定；命令行训练配置提供 standard、RandAugment 和 AutoAugment 选项。

### 4. 组合预处理、集成和评估

```python
from aegis_lab.preprocessing import preprocess
from aegis_lab.ensembles import Ensemble
from aegis_lab.evaluation import evaluate

filtered = preprocess(images, [
    {"kind": "bit_depth", "bits": 5},
    {"kind": "median", "kernel": 3},
])

ensemble = Ensemble([model_a, model_b], method="soft_vote").eval()
logits = ensemble(filtered)

results = evaluate(model, test_loader, [
    {"method": "pgd20", "epsilon_255": 8},
    {"method": "cw_linf", "epsilon_255": 8},
])
```

集成成员采用相同的输入格式与类别顺序。更多示例见 [docs/API.md](docs/API.md)。

## 模型与数据集

内置模型结构：

| 类型 | 名称 |
| --- | --- |
| CIFAR 适配 ResNet | `cifar_resnet18`、`cifar_resnet34` |
| torchvision 网络 | `resnet18`、`resnet34`、`densenet121`、`mobilenet_v2`、`efficientnet_b0`、`wide_resnet50_2` |
| 自定义模型 | Python API 接收 `torch.nn.Module`；工作台支持加载工厂与权重 |

CIFAR 适配 ResNet 使用 3×3 首层卷积并移除初始最大池化。

支持 CIFAR-10、CIFAR-100 与 ImageFolder。自定义图像分类数据使用目录名作为类别名：

```text
dataset/
├── class_a/
│   ├── image_001.png
│   └── image_002.png
└── class_b/
    ├── image_001.png
    └── image_002.png
```

ImageFolder 训练在配置中指定 `dataset="imagefolder"`、`data_dir`、`mean`、`std`；可以通过 `validation_dir` 指定独立验证目录。

## 实验复现

统一配置位于 [configs/experiment-plan.json](configs/experiment-plan.json)。以下数量表示默认实验计划，训练与评测结果由实际运行生成。

### 默认训练矩阵

| 实验组 | 组合 | 任务数 |
| --- | --- | ---: |
| 常规分类 | 2 数据集 × 3 架构 × ERM / PGD-AT / TRADES | 18 |
| 防御蒸馏 | 2 数据集 × CIFAR ResNet-18 × 教师 / 学生 | 4 |
| 长尾鲁棒性 | 2 数据集 × CIFAR ResNet-18 × IR10 / IR100 × 5 方法 | 20 |
| **合计** | 默认 seed=0，100 epochs/任务 | **42** |

两个数据集为 CIFAR-10/100；三个主架构为 CIFAR ResNet-18、DenseNet-121、MobileNetV2。长尾组使用 ERM、PGD-AT、TRADES、AT-BSL、TAET。AT-BSL 配置 RandAugment；TAET 先进行 40 轮干净预训练。

### 评估矩阵

| 内容 | 默认设置 |
| --- | --- |
| 攻击 | FGSM、PGD20、PGD100、CW-L∞、DeepFool-L∞，ε=2/4/8；AA standard，ε=8 |
| 每模型攻击参数组 | 16 组 |
| 测试集 | 官方完整测试集；`sample_limit=0` |
| 检测 | 两数据集的 ResNet-18 ERM / PGD-AT，共 4 模型 × 6 检测器 |
| 检测校准 | 目标 FPR=5%，使用独立干净校准子集 |
| 检测自适应攻击 | 分类损失与检测规避联合目标的 PGD |
| 输入处理与集成 | 5 种输入处理、4 种集成策略，分别评估迁移与自适应攻击 |
| 异常输入 | 跨 CIFAR 数据、噪声、模糊、亮度变换 |
| 输出监测 | 固定窗口的输出概率分布偏移 |

### 分阶段执行

```bash
# 仅展开并查看任务
python -m aegis_lab experiment --stage plan

# 训练常规模型、蒸馏模型和长尾模型
python -m aegis_lab experiment --stage train --group main --download
python -m aegis_lab experiment --stage train --group long_tail --download

# 运行评估
python -m aegis_lab experiment --stage evaluate
python -m aegis_lab experiment --stage detect --group main
python -m aegis_lab experiment --stage defend --group main
python -m aegis_lab experiment --stage anomaly --group main

# 导出成果
python -m aegis_lab export --include-weights
```

也可在数据准备好后使用 `bash scripts/run-experiments.sh` 顺序执行。服务器部署、日志管理与 DSH 提示词见 [DSH_SERVER_HANDOFF.md](DSH_SERVER_HANDOFF.md)。

常用选项：

| 参数 | 用途 |
| --- | --- |
| `--plan 路径` | 使用指定 JSON 实验计划 |
| `--group main` / `long_tail` / `all` | 选择实验组 |
| `--only 字符串` | 按任务 ID 子串筛选 |
| `--download` | 允许下载标准数据集 |
| `--smoke` | 独立目录内运行两轮、小样本流程检查 |

例如，只训练一个小样本任务：

```bash
python -m aegis_lab experiment --stage train \
  --only cifar10-cifar_resnet18-erm-ir1-s0 --smoke --download
```

训练任务已有 `last.pt` 时，重复相同命令会继续训练。单个配置文件也可以独立运行：

```bash
python -m aegis_lab train --config configs/cifar10-at-bsl.json
python -m aegis_lab train --config configs/cifar10-at-bsl.json --resume 任务目录/last.pt
```

### 产物结构

```text
data/experiments-v04/<任务ID>/
├── config.json                  # 实际训练配置
├── split.json                   # 训练 / 验证索引及类别数量
├── evaluation_split.json        # 检测拟合、校准和测试索引
├── last.pt                      # 训练恢复检查点
├── best_weights.pt              # 选定模型权重
├── model_registration.json      # 模型结构与预处理信息
├── metrics.jsonl                # 训练与验证曲线
├── run.json                     # 任务、命令和依赖信息
├── evaluation.json              # 攻击评估指标
├── detection.json               # 检测器评估
├── defenses.json                # 预处理 / 集成评估
├── anomaly.json                 # 异常评估与窗口事件
├── cases/                       # 原图、对抗图与扰动图
├── detectors/                   # 检测器参数与阈值
└── anomaly/                     # 异常检测器与窗口参数
```

小样本流程输出到 `data/experiments-v04-smoke/`。

## 成果网站

网站采用 Vue 3 与 Vite，主导航为：

**项目介绍｜方法与功能｜实验成果｜模型与下载｜使用文档**

- **项目介绍**：项目定位、功能覆盖、快速开始与实验入口。
- **方法与功能**：方法原理、调用代码、参数与论文参考。
- **实验成果**：按数据集、模型、方法、IR 和种子筛选；比较预算曲线、逐类表现、检测 ROC、训练曲线与案例。
- **模型与下载**：模型结构、数据介绍、模型卡、配置、权重与报告。
- **使用文档**：安装、快速开始、六类功能教程、API 与完整复现流程。

### 启动与构建

```bash
npm ci
npm run dev
```

开发地址：[http://127.0.0.1:5173/](http://127.0.0.1:5173/)。

```bash
npm run build
npm run preview
```

构建目录为 `dist/`。网站使用相对资源路径与 hash 路由，可将构建产物放到静态托管服务。公开成果与文档页面直接读取静态文件。

### 更新实验结果

```bash
python -m aegis_lab export \
  --source data/experiments-v04 \
  --destination public/data \
  --include-weights

npm run build
```

导出器生成 `results.json`，并将配置、案例、检测器、CSV 与 HTML 报告放入 `downloads/`。`--include-weights` 同时复制模型权重。网站下载项也支持 HTTP(S) URL，便于使用独立模型文件托管。

### 可选工作台

在项目根目录启动后端：

```bash
python -m pip install -e ".[attacks,workbench]"
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

另一个终端运行 `npm run dev`，打开：

- 工作台：[http://127.0.0.1:5173/#/workbench/overview](http://127.0.0.1:5173/#/workbench/overview)
- 后端 API 文档：[http://127.0.0.1:8765/docs](http://127.0.0.1:8765/docs)

工作台通过 `/api` 调用本机后端，提供模型接入、数据登记、训练、攻防配置和实验记录。公开在线体验入口后续接入。

## 工程结构

```text
AI-Safety-Lab/
├── aegis_lab/                   # 可安装 Python 算法库
│   ├── attacks.py              # 统一攻击接口
│   ├── detection.py            # 攻击检测器
│   ├── anomaly.py              # 输入 / 输出异常检测
│   ├── training.py             # DataLoader 训练接口
│   ├── preprocessing.py        # 输入变换
│   ├── ensembles.py            # 集成预测
│   ├── evaluation.py           # 指标与评估
│   ├── experiments.py          # 实验矩阵与阶段执行
│   ├── publishing.py           # 静态成果导出
│   ├── cli.py                  # 命令行
│   └── _core/                  # 攻击、训练、数据与模型核心实现
├── backend/                    # 可选工作台 API 与兼容入口
├── src/                        # Vue 网站与工作台
├── public/                     # 静态资源与导出成果
├── configs/                    # 实验计划与训练配置
├── scripts/                    # 服务器执行脚本
├── examples/                   # 接入示例
├── tests/                      # 算法与接口测试
├── docs/API.md                 # 详细接口与算法配置
├── pyproject.toml              # Python 安装配置
├── package.json                # 前端依赖与脚本
├── DSH_SERVER_HANDOFF.md        # 服务器任务交接
└── THIRD_PARTY_NOTICES.md       # 方法来源与第三方依赖
```

## 测试与开发

```bash
python -m pip install -e ".[attacks,workbench]"
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
npm run build
```

测试覆盖攻击公式与预算、检测器拟合与保存恢复、Mahalanobis 数值、Layer Regression 优化、输出漂移窗口、实验计划展开、导出文件以及工作台接口。

最近记录：68 项 CPU 测试通过、前端生产构建通过、独立 wheel 导入通过。GPU 实验结果将在服务器运行后导出，详细记录见 [VALIDATION.md](VALIDATION.md)。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [API 与算法说明](docs/API.md) | 参数、算法公式、配置选择与使用示例 |
| [实验计划](configs/experiment-plan.json) | 数据集、模型、方法和评估组合 |
| [服务器运行说明](DSH_SERVER_HANDOFF.md) | GPU 启动顺序、日志、产物与 DSH 提示词 |
| [迁移说明](MIGRATION_GUIDE.md) | 复制工程与服务器安装 |
| [实现对应表](IMPLEMENTATION_STATUS.md) | 功能、代码、实验与展示对应 |
| [验证记录](VALIDATION.md) | 已执行检查 |
| [第三方来源](THIRD_PARTY_NOTICES.md) | 参考方法与依赖说明 |

## 参考项目与论文

- [Foolbox](https://github.com/bethgelab/foolbox)：DeepFool 攻击实现。
- [AutoAttack](https://github.com/fra31/auto-attack)：standard 组合攻击。
- [TRADES](https://github.com/yaodongyu/TRADES)：鲁棒训练目标。
- [AT-BSL](https://github.com/NISPLab/AT-BSL)：长尾对抗训练与 Balanced Softmax。
- [TAET](https://github.com/Eureka-WangYuHang/TAET-Two-Stage-Adversarial-Equalization-Training-on-Long-Tailed-Distributions)：两阶段长尾对抗训练。
- [Feature Squeezing](https://arxiv.org/abs/1704.01155)：通过输入压缩与预测差异检测对抗样本。
- [Mahalanobis Detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector)：类条件高斯特征检测。
- [Layer Regression](https://github.com/furkanmumcu/Layer-Regression)：通过层间特征回归检测对抗输入。
- [Energy-based OOD Detection](https://arxiv.org/abs/2010.03759)：基于能量分数的分布外检测。

本项目的方法配置与集成方式见 API 文档；第三方依赖沿用各自许可证与来源说明。
