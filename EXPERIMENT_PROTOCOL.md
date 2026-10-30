# 攻击与批量实验协议 · 0.3.0

本文件描述实际实现，不是论文结果或 CUDA 验收记录。

## 当前攻击

| 接口名称 | 方法 | 预算与主要参数 |
|---|---|---|
| `pgd20` | 交叉熵 PGD，固定 20 步 | L∞；步长与重启显式设置 |
| `pgd100` | 交叉熵 PGD，固定 100 步 | L∞；步长与重启显式设置 |
| `cw_linf` | CW margin 损失 + 随机起点 + L∞ 投影 | 默认 100 步；不是经典 C&W-L2 |
| `autoattack` | 官方 AutoAttack `standard` | L∞，不修改组件与内部参数 |
| `deepfool_linf` | Foolbox `LinfDeepFoolAttack` | L∞；最大步数、候选类别与 overshoot 可配置 |
| `deepfool_l2` | Foolbox `L2DeepFoolAttack` | L2；单独分组 |
| `fgsm` / `pgd` | FGSM / 自定义步数 PGD | 兼容原有单图接口 |

PGD-20 和 PGD-100 是同一算法的两个固定实验预设，不包装成两种新算法。预算始终施加在归一化之前的 RGB `[0,1]` 像素空间。L∞ 的界面单位为 `/255`，L2 直接使用像素空间的 L2 数值。

## CW 的定义和待对齐项

所提供 LAAC 论文第 3.1 节明确报告 L∞ 预算 8/255、PGD 20/100 步且步长 1/255，以及 100 步 CW；这些明确参数已对应到默认配置。按用户决定，CW 以 L∞ 为主。论文正文未充分给出 CW 损失和所有实现细节，当前使用下述损失最大化：

`-max(z_true - max(z_other), -kappa)`

攻击在 L∞ 球内迭代投影。默认预算 8/255、100 步、步长 1/255、kappa=0、1 次随机起点，均写入结果协议。损失是否截断、置信参数、步长、随机起点和重启次数还需要与原实验代码逐项核对；不能据此称作论文 CW 的完全复现。

## AutoAttack 来源

- 官方项目：https://github.com/fra31/auto-attack
- 固定提交：`a39220048b3c9f2cca9a4d3a54604793c68eca7e`
- 版本：`standard`
- 组件：`apgd-ce`, `apgd-t`, `fab-t`, `square`

运行时核对组件列表，并记录各组件的迭代、目标类别或查询数。不减少查询预算或删掉组件后继续称其为完整 AA。官方会对已被早期组件攻破的样本跳过后续计算，这属于 standard 自身流程。

本接口保留标准的 9 个目标类别，因此要求至少 10 类模型；低类别数模型会明确拒绝，不悄悄改成自定义套件。带随机化或不可微防御的模型不应直接套用本确定性 standard 协议，后续防御模块需要单独做自适应评估。

零预算直接返回原输入，并记录跳过攻击计算，不伪造迭代过程。

## 数据与模型接入

- 内置只读 CIFAR-10/100 官方测试集，不自动下载。
- 自定义 ImageFolder 按目录名排序建立类别编号；与模型类别名称、顺序核对。
- 验证集可登记为独立 ImageFolder。测试集和验证集名称与划分写入报告。
- 数据目录检查不需要 GPU；真正评测默认要求 CUDA。

目录指纹包含相对路径、标签、大小和修改时间，不等同于逐图片内容哈希。实际执行还保存每一批经过确定性缩放和张量转换后的输入 SHA256。

## 批量运行与恢复

模型 × 攻击 × 对应范数预算形成实验矩阵。所有组合使用相同样本索引；`sample_limit=0` 使用完整划分，正数为固定种子的随机子集，报告明确标注实际样本数。

批次随机种子为 `base_seed + batch_index`。单模型、范数和预算的联合鲁棒率逐样本合并各攻击失败的并集，不平均攻击准确率，不混合 L∞ 和 L2。

每完成一个批次原子写入 JSON 检查点。恢复必须同时匹配配置、模型/加载文件摘要、数据目录指纹、依赖与代码版本，并逐批复核实际输入摘要。正在执行的批次会重新计算；不承诺在 AA 单个内部迭代的位置恢复。

同一组合耗时为各批真实计算耗时之和，包含攻击初始化和推理。复用检查点时仍保留原有计算耗时，并单独记录复用批次数；它不是本次恢复等待时间，也不是纯 GPU 内核时间。

## 输出

- `manifest.json`：协议、资源与版本摘要。
- `sample_indices.json`：实际样本索引。
- `report.json`：各组合、逐类与联合指标。
- `summary.csv`：组合汇总。
- `per_class.csv`：逐类别结果。
- `samples.csv`：逐样本预测和扰动。
- 各组合目录：批次检查点、成功案例预览，以及 AA 日志。

网页可查看曲线、表格、逐类别指标和成功案例。没有真实结果时显示空状态，不填入示范准确率。

## 服务器运行

先安装与显卡和驱动匹配的 GPU 版 PyTorch，再运行：

```bash
python -m pip install -r requirements-attacks.txt
python -m backend.evaluate_suite --config configs/evaluation-cifar10.json --output-dir data/evaluation/full-suite --check-only
python -m backend.evaluate_suite --config configs/evaluation-cifar10.json --output-dir data/evaluation/full-suite
```

配置中的数据根目录和权重路径需要按服务器实际情况修改。先用单独配置的少量样本做服务器验收；正式实验使用新输出目录及 `sample_limit=0`。恢复相同实验时使用原配置和原输出目录。

本机只做临时小网络功能检查，不运行正式训练，不把本机检查视为服务器完整实验。

## 0.3.0 扩展

防御、检测、蒸馏、自适应攻击及实时接口另见 `DEFENSE_PROTOCOL.md`。AA 安装来源会核对固定官方提交，依赖摘要扩充到 TorchVision、NumPy、Pillow 与 SciPy。
