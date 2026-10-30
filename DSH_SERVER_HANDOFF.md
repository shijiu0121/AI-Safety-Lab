# DSH 服务器运行说明 · Aegis Lab 0.4.0

## 发给 DSH 的提示词

```text
请在当前工程目录阅读 AGENTS.md、README.md、DSH_SERVER_HANDOFF.md、docs/API.md 和 configs/experiment-plan.json，按本文完成 GPU 小样本检查，然后启动正式实验。工程版本是 0.4.0。

先确认项目实际路径、Python、nvidia-smi、GPU 名称和可用显存，使用项目专用环境。用户描述的 GPU 是 5880，以实际设备为准。安装匹配 CUDA 的 PyTorch/TorchVision，再 pip install -e ".[attacks]"。下载或复用 CIFAR-10 和 CIFAR-100。

先执行 plan 查看 42 个训练任务，再执行本文的小样本检查。随后按阶段训练、评估、导出，使用 tmux 或 nohup 保存日志。训练和测试使用同一份 experiment-plan.json。正常步骤直接推进；遇到数据路径不明、显存被其他任务占用等具体问题再询问用户。不要改动其他项目。

任务分为：常规组 22 个模型，长尾组 20 个模型；随后进行攻击、检测、预处理/集成、异常检测实验，最后导出公开成果页。初次运行可以先启动 main 组；本计划包括两组的全部实验。记录每个阶段的日志、开始时间、PID/tmux 名、完成状态和产物路径。

使用下方命令，现有 last.pt 可继续训练。若修改训练参数，请使用新的 output_dir。网站读 public/data/results.json，导出后运行 npm run build。后台正式任务启动后报告 GPU、任务数量、当前阶段、日志路径与进程是否正常，不需要在当前会话一直等待所有实验结束。
```

## 实验数量

| 实验组 | 内容 | 训练任务 |
| --- | --- | ---: |
| 常规分类 | 2 数据集 × 3 架构 × ERM/PGD-AT/TRADES | 18 |
| 防御蒸馏 | 2 数据集 × ResNet-18 × 高温教师/学生 | 4 |
| 长尾 | 2 数据集 × ResNet-18 × IR10/IR100 × ERM/PGD-AT/TRADES/AT-BSL/TAET | 20 |
| 合计 | 默认 seed=0，100 epochs/任务 | 42 |

3 个主架构为 cifar_resnet18、densenet121、mobilenet_v2。AT-BSL 使用 RandAugment；TAET 前 40 轮干净预训练。

攻击评测：FGSM、PGD20、PGD100、CW-L∞、DeepFool-L∞ 分别在 ε=2/4/8 下运行，AA standard 在 ε=8 下运行，共 16 组/模型。完整测试集 10,000 张。默认约 672 个“模型 × 攻击配置”评测单元。

检测：CIFAR-10/100 的 ResNet-18 ERM、PGD-AT 共 4 个模型，每个比较 6 个检测器，共 24 个检测器配置。每个用 ε=8 的六种攻击与一组联合分类/检测规避 PGD 测试。Mahalanobis 默认末层共享协方差距离，Layer Regression 隐藏层宽度 256、训练 10 轮；详细设置在配置和 API 文档。

输入处理与集成：每个数据集的 PGD-AT ResNet-18 上比较 5 种输入处理；使用 3 个 PGD-AT 架构组合 4 种投票策略；分别评估迁移和自适应攻击，共 36 个策略评测单元。

异常：4 个模型分别比较 InputStatistics、Energy、Mahalanobis；跨 CIFAR 分布外输入及噪声、模糊、亮度异常，另保存 OutputDrift 窗口事件。

## 运行命令

所有命令在复制后的工程根目录执行。

```bash
python -m pip install -e ".[attacks]"
aegis experiment --stage plan > experiment-plan-expanded.json

# 两个数据集都准备好
python -c "from torchvision.datasets import CIFAR10,CIFAR100; [c('data/datasets',train=t,download=True) for c in (CIFAR10,CIFAR100) for t in (True,False)]"

# 单模型 GPU 小样本检查：训练、攻击、检测和异常
# --only 为精确 ID，避免同时选中长尾或其他方法
RUN=cifar10-cifar_resnet18-erm-ir1-s0
aegis experiment --stage train --only "$RUN" --smoke
aegis experiment --stage evaluate --only "$RUN" --smoke
aegis experiment --stage detect --only "$RUN" --smoke
aegis experiment --stage anomaly --only "$RUN" --smoke

# 正式流水线
mkdir -p data/launch-logs
bash scripts/run-experiments.sh
```

`run-experiments.sh` 顺序执行全部阶段；建议在 tmux 中启动。每个阶段的日志保存在 `data/launch-logs/`。也可以单独执行：

```bash
aegis experiment --stage train --group main
aegis experiment --stage train --group long_tail
aegis experiment --stage evaluate
aegis experiment --stage detect --group main
aegis experiment --stage defend --group main
aegis experiment --stage anomaly --group main
aegis export --include-weights
```

默认正式输出 `data/experiments-v04/`，小样本输出 `data/experiments-v04-smoke/`。恢复训练重复相同 train 命令，自动读取各任务的 last.pt。评测可用 --only 逐任务重跑。

GPU 自适应攻击检查还需至少运行一个 PGD-AT 模型的 `defend` 阶段；该阶段的集成依赖同数据集 3 个架构的 PGD-AT 权重。可以先在 smoke 目录训练这些任务，再运行 defend。

## 产物与网站

每个任务目录包含：config.json、split.json、last.pt、best_weights.pt、model_registration.json、metrics.jsonl、run.json。
评测后增加 evaluation_split.json、evaluation.json、cases/、detection.json、detectors/、defenses.json、anomaly.json、anomaly/。

```bash
aegis export --include-weights
npm install
npm run build
```

导出生成 `public/data/results.json`、`downloads/` 下的配置、模型权重、检测器、CSV、HTML 报告和图片。浏览器可直接比较与下载。复制成果回本机后，也可以重新执行 export（图片按任务目录内相对位置定位）。

需要公开托管大型权重时，可将其上传到模型托管服务，再把 results.json 的 downloads.weights 改为下载 URL；网页支持直接下载 URL。

## 进度记录

写入 `data/launch-logs/DSH_RUN_STATUS.md`：工程与 Python 路径、GPU、计划文件、后台命令与 PID、当前阶段、完成任务 ID、日志路径、实际错误及处理。正式评测数据由程序输出，CPU 测试记录见 VALIDATION.md。
