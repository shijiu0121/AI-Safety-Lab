# Aegis Lab 0.4 · Python API 与实验教程

## 1. 模型与输入

```python
from aegis_lab import Classifier, create_model
net = create_model('cifar_resnet18', num_classes=10)
model = Classifier(net, mean=(.4914,.4822,.4465), std=(.247,.2435,.2616)).cuda().eval()
```

模型输出 `[N,C]` logits；图像为 `[N,3,H,W]`、float32、像素范围 [0,1]。用自己的 `torch.nn.Module` 替换 net 即可。归一化由 Classifier 执行。库函数沿用模型/张量所在设备；CLI 训练使用 CUDA。

内置结构：cifar_resnet18/34、resnet18/34、densenet121、mobilenet_v2、efficientnet_b0、wide_resnet50_2。数据读取支持 CIFAR-10/100、ImageFolder；统一矩阵使用两种 CIFAR。

## 2. 攻击

```python
from aegis_lab import attack
adversarial, detail = attack(model, images, labels, 'pgd100',
    epsilon_255=8, step_size_255=2, restarts=1, return_details=True)
```

method：fgsm、pgd、pgd20、pgd100、cw_linf、deepfool_linf、deepfool_l2、autoattack。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| epsilon_255 | 8 | L∞ 预算，实际值为该数 /255 |
| step_size_255 | 1 | 每步像素值增量；正式计划为 2 |
| steps | 20 | 自定义 pgd 步数；pgd20/100 固定步数 |
| restarts | 1 | PGD/CW 随机重启次数 |
| cw_steps | 100 | CW-L∞ 迭代数 |
| cw_confidence | 0 | margin 损失置信参数 |
| epsilon_l2 | 0.5 | DeepFool-L2 预算 |
| deepfool_steps | 50 | DeepFool 最大步数 |
| deepfool_candidates | 10 | DeepFool 候选类数 |
| seed | 0 | 随机种子 |

CW-L∞ 为最大化 `-max(z_y-max(z_other),-kappa)` 的投影迭代。DeepFool 调用 Foolbox 3.3.4；AA 调用固定官方提交的 standard 组合。detail 包含攻击协议、迭代记录和依赖版本。攻击临时切换 eval，结束后恢复各层原训练状态。

## 3. 检测与异常

```python
from aegis_lab.detection import FeatureSqueezing, Mahalanobis, LayerRegression, load_detector

squeeze = FeatureSqueezing(model, bits=5, kernel=3)
maha = Mahalanobis(model, layers=['model.avgpool'], regularization=1e-4)
lr = LayerRegression(model, layers=['model.layer2','model.layer3'],
    target_layer='model.avgpool', hidden_dim=256)

maha.fit(fit_loader)
maha.calibrate(calibration_loader, false_positive_rate=.05)
scores = maha.score(images)
flags = maha.predict(images)
maha.save('maha.pt')
maha = load_detector('maha.pt', model)

lr.fit(fit_loader, epochs=10, learning_rate=.0003)
lr.calibrate(calibration_loader)
```

所有分数越大表示越异常。`fit` 拟合分布或回归器，`calibrate` 使用独立干净验证数据的分位数设置阈值。Confidence、Entropy、Consistency、FeatureSqueezing 无需拟合参数，可直接校准。

- Confidence：`1-max(softmax(logits))`。
- Entropy：预测熵除以 `log(类别数)`。
- Consistency：原图与中值滤波预测概率的半 L1 距离。
- Feature Squeezing：分别进行位深压缩与中值滤波，取两种概率 L1 距离的最大值；默认 bits=5、kernel=3。选择了原论文的位深与平滑联合检测配置。
- Mahalanobis：空间平均后的特征，每类一个均值、共享协方差，分数为最小半平方马氏距离；多层默认取平均。可选 `magnitude` 做输入梯度预处理，默认 0。`fit_fusion(clean_images, adversarial_images)` 可在独立开发集上拟合逻辑回归融合器。默认实验使用单个末层特征与干净集校准。
- Layer Regression：按用户给定的模块名取特征，展平后切取中间 60%，使用两隐藏层 ReLU MLP 回归末层的空间平均特征。只用分类正确的干净样本训练。公开接口默认 hidden_dim=2048；本项目矩阵显式使用 256。层名、切片、宽度、拟合轮数随检测器保存，采用固定层配置。

```python
from aegis_lab.anomaly import InputStatistics, Energy, OutputDrift
statistics = InputStatistics(model).fit(fit_loader).calibrate(calibration_loader)
energy = Energy(model, temperature=1).calibrate(calibration_loader)
monitor = OutputDrift(window_size=128).fit(reference_probabilities)
monitor.calibrate(calibration_probabilities)
events = monitor.update(new_probabilities)
monitor.save('drift.pt')
monitor = OutputDrift.load('drift.pt')
```

InputStatistics 使用每通道均值、标准差、饱和率与竖直相邻像素绝对差，经训练集标准化后取最大绝对 z 分数。Energy 为 `-T*logsumexp(logits/T)`。OutputDrift 以固定窗口平均概率和参考平均概率计算 Jensen-Shannon 距离，校准数据按相同窗口切分。样本不足一个窗口时暂存，凑够后返回窗口事件。

参考：[Feature Squeezing](https://arxiv.org/abs/1704.01155)、[Mahalanobis](https://github.com/pokaxpoka/deep_Mahalanobis_detector)、[Layer Regression](https://github.com/furkanmumcu/Layer-Regression)、[Energy](https://arxiv.org/abs/2010.03759)。本工程按方法公式与公开实现逻辑重新实现，并记录选用配置。

## 4. 鲁棒性训练

```python
from aegis_lab.training import train_epoch
loss = train_epoch(model, train_loader, optimizer,
    method='at_bsl', epoch=0, num_classes=10,
    class_counts=class_counts, attack_steps=10,
    epsilon_255=8, step_size_255=2)
```

`train_epoch` 接受用户已有 DataLoader 与 optimizer；`epoch` 从 0 开始，返回平均训练损失。

| method | 训练目标 |
| --- | --- |
| erm | 干净样本交叉熵 |
| pgd_at | PGD 对抗样本交叉熵 |
| trades | 干净 CE + beta × 对抗/干净预测 KL |
| at_bsl | 对抗样本 `CE(logits+log(class_counts),labels)` |
| taet | 先干净 CE，之后按类别均值、方差和归一化平方和构成 HARL |
| distill_teacher | 温度缩放后的教师交叉熵 |
| distill | 硬标签 CE 与教师软标签 KL，KL 乘 T² |

TAET 参数 `taet_warmup_epochs=40, taet_alpha=taet_beta=taet_gamma=.1`。AT-BSL 的 class_counts 来自训练子集。数据增强由 DataLoader 决定，CLI 可配置 standard/randaugment/autoaugment。

蒸馏接口传入 `teacher=teacher_model`。教师与学生均采用接收 [0,1] 图像的 Classifier。普通蒸馏使用常规教师；防御蒸馏先训练高温教师，再训练学生。教师 softmax 温度、学生温度和硬标签权重见 temperature、distill_alpha。

CLI：`aegis train --config configs/cifar10-at-bsl.json`；恢复：`aegis train --config 原配置 --resume 任务目录/last.pt`。

## 5. 输入处理与集成

```python
from aegis_lab.preprocessing import preprocess
from aegis_lab.ensembles import Ensemble
filtered = preprocess(images, [{'kind':'median','kernel':3}])
ensemble = Ensemble([model_a,model_b], method='soft_vote',
    transforms=[{'kind':'noise','noise_std':.01}], samples=4)
logits = ensemble(images)
```

变换 kind：bit_depth、median、gaussian、resize_pad、noise。投票 method：soft_vote、logit_mean、hard_vote、random_member。成员均需相同类别顺序。

自适应攻击设置 `surrogate=True`：量化/中值滤波用 BPDA，硬投票攻击用软投票代理，随机模型与变换平均多次预测。最终评测使用 `surrogate=False`。

## 6. 评估、统一计划与成果导出

```python
from aegis_lab.evaluation import evaluate, evaluate_detector
rows = evaluate(model, test_loader, [
    {'method':'pgd20','epsilon_255':8},
    {'method':'cw_linf','epsilon_255':8}])
detection = evaluate_detector(detector, test_loader,
    {'method':'pgd100','epsilon_255':8})
adaptive = evaluate_detector(detector, test_loader,
    {'method':'adaptive_pgd','epsilon_255':8,'steps':100}, adaptive=True)
```

鲁棒准确率为原图和对抗图均分类正确的比例；attack_success_rate 的分母为原图分类正确数。检测指标报告全部生成的攻击输入；规避成功率报告原图正确、攻击后错误且未触发检测的比例。自适应目标为分类 CE 减去检测超阈值惩罚，可调 weight。

正式计划 `configs/experiment-plan.json`。训练集先按类别分出 10% 验证集，再进行长尾采样；训练模型的验证子集再分成检测开发与阈值校准两部分。检测器 fit 使用训练子集，分类与检测测试使用官方测试集。划分索引保存为 split.json 与 evaluation_split.json。模型检查点以验证 PGD 准确率选择。

异常评估使用 CIFAR-10/100 相互作为 OOD 数据，另比较高斯噪声 std=.1、高斯模糊 sigma=1、亮度 +.25。集成使用同数据集三个 PGD-AT 模型，默认四次随机采样。

```bash
aegis experiment --stage plan
aegis experiment --stage train --group main
aegis experiment --stage train --group long_tail
aegis experiment --stage evaluate
aegis experiment --stage detect --group main
aegis experiment --stage defend --group main
aegis experiment --stage anomaly --group main
aegis export --source data/experiments-v04 --destination public/data --include-weights
```

导出为 results.json + downloads/，支持静态网站。JSON 内包含每个模型的配置、依赖版本、训练曲线、评测结果与案例路径；报告提供 CSV 与 HTML。新跑实验后重复 export，再构建网站即可。
