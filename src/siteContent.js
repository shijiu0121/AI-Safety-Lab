export const methods = [
  {
    id: "attacks",
    name: "对抗样本生成",
    subtitle: "从单步扰动到组合攻击",
    items: [
      "FGSM",
      "PGD / PGD-20 / PGD-100",
      "CW-L∞",
      "DeepFool-L∞ / L2",
      "AutoAttack standard",
    ],
    description:
      "梯度攻击在扰动预算内寻找误分类输入；DeepFool 迭代逼近决策边界；AutoAttack 组合 APGD、FAB 和 Square。CW-L∞ 使用 CW margin 损失与 L∞ 投影迭代。",
    params: [
      ["epsilon_255", "8", "L∞ 预算，除以 255 后用于 [0,1] 图像"],
      ["step_size_255", "2", "迭代步长"],
      ["restarts", "1", "随机重启次数"],
      ["epsilon_l2", "0.5", "DeepFool-L2 预算"],
    ],
    code: 'from aegis_lab import attack\n\nx_adv = attack(model, images, labels,\n    method="pgd100", epsilon_255=8,\n    step_size_255=2, restarts=1)',
    refs: [
      ["Foolbox", "https://github.com/bethgelab/foolbox"],
      ["AutoAttack", "https://github.com/fra31/auto-attack"],
    ],
  },
  {
    id: "detection",
    name: "攻击检测",
    subtitle: "评分、校准与检测器复用",
    items: [
      "置信度 / 预测熵",
      "变换一致性",
      "Feature Squeezing",
      "Mahalanobis",
      "Layer Regression",
    ],
    description:
      "Feature Squeezing 比较量化、平滑前后的预测差异。Mahalanobis 估计类条件特征均值与共享协方差。Layer Regression 用中间层特征回归深层特征，以预测误差检测异常输入。",
    params: [
      ["false_positive_rate", "0.05", "干净校准集目标误报率"],
      ["layers", "模块名列表", "中间特征层"],
      ["target_layer", "model.avgpool", "回归目标层"],
      ["hidden_dim", "2048", "回归网络维度；实验计划使用 256"],
      ["regularization", "1e-4", "协方差正则项"],
    ],
    code: 'from aegis_lab.detection import Mahalanobis\n\ndetector = Mahalanobis(model, ["model.avgpool"])\ndetector.fit(train_loader)\ndetector.calibrate(calibration_loader,\n    false_positive_rate=0.05)\nflags = detector.predict(images)\ndetector.save("detector.pt")',
    refs: [
      ["Feature Squeezing · NDSS 2018", "https://arxiv.org/abs/1704.01155"],
      ["Mahalanobis · NeurIPS 2018", "https://arxiv.org/abs/1807.03888"],
      [
        "Layer Regression · TMLR 2025",
        "https://github.com/furkanmumcu/Layer-Regression",
      ],
    ],
  },
  {
    id: "training",
    name: "鲁棒性训练",
    subtitle: "常规分类、长尾学习与蒸馏",
    items: [
      "ERM",
      "PGD-AT / TRADES",
      "AT-BSL / TAET",
      "高温教师训练",
      "防御蒸馏 / 知识蒸馏",
    ],
    description:
      "PGD-AT 使用对抗样本训练；TRADES 平衡分类损失与预测分布差异。AT-BSL 加入类别频次修正；TAET 先进行干净预训练，再进行类别损失均衡的对抗训练。蒸馏使用教师软标签指导学生。",
    params: [
      [
        "method",
        "pgd_at",
        "erm / pgd_at / trades / at_bsl / taet / distill_teacher / distill",
      ],
      ["trades_beta", "6", "鲁棒项权重"],
      ["imbalance_ratio", "1", "头尾类别目标样本比"],
      ["taet_warmup_epochs", "40", "干净预训练轮数"],
      ["temperature", "20", "蒸馏温度"],
      ["distill_alpha", "0.5", "硬标签损失权重"],
    ],
    code: 'from aegis_lab.training import train_epoch\n\nloss = train_epoch(model, train_loader, optimizer,\n    method="at_bsl", epoch=0, num_classes=10,\n    class_counts=counts, attack_steps=10)',
    refs: [
      ["TRADES", "https://github.com/yaodongyu/TRADES"],
      ["AT-BSL · CVPR 2024", "https://github.com/NISPLab/AT-BSL"],
      [
        "TAET · CVPR 2025",
        "https://github.com/Eureka-WangYuHang/TAET-Two-Stage-Adversarial-Equalization-Training-on-Long-Tailed-Distributions",
      ],
    ],
  },
  {
    id: "preprocessing",
    name: "输入预处理",
    subtitle: "可组合的图像变换",
    items: ["位深压缩", "中值 / 高斯滤波", "缩放填充", "随机噪声"],
    description:
      "把量化、滤波与尺寸变换串联到预测流程。评估量化和中值滤波时可使用 BPDA 梯度近似；随机变换使用多次采样。",
    params: [
      ["bits", "5", "每通道保留位数"],
      ["kernel", "3", "滤波核尺寸"],
      ["sigma", "0.5", "高斯核标准差"],
      ["scale", "0.9", "缩放比例"],
      ["noise_std", "0.01", "噪声标准差"],
    ],
    code: 'from aegis_lab.preprocessing import preprocess\n\nprocessed = preprocess(images, [\n    {"kind": "bit_depth", "bits": 5},\n    {"kind": "median", "kernel": 3}\n])',
    refs: [],
  },
  {
    id: "ensembles",
    name: "模型集成",
    subtitle: "多模型共同预测",
    items: ["软投票", "logits 平均", "硬投票", "随机成员 / 多次采样"],
    description:
      "软投票聚合概率，logits 平均在 softmax 前聚合，硬投票统计类别票数。随机成员从模型列表中抽取成员，多次预测后平均。",
    params: [
      [
        "method",
        "soft_vote",
        "soft_vote / logit_mean / hard_vote / random_member",
      ],
      ["samples", "4", "随机策略采样次数"],
      ["transforms", "[]", "输入变换列表"],
      ["surrogate", "False", "攻击时启用 BPDA 和软投票代理"],
    ],
    code: 'from aegis_lab.ensembles import Ensemble\n\nensemble = Ensemble([resnet, densenet, mobile],\n    method="soft_vote")\nlogits = ensemble(images)',
    refs: [],
  },
  {
    id: "anomaly",
    name: "异常检测",
    subtitle: "输入、输出与数据流",
    items: [
      "输入统计",
      "Energy 分布外检测",
      "Mahalanobis 特征距离",
      "OutputDrift 窗口偏移",
    ],
    description:
      "输入统计比较亮度、方差、饱和率与相邻像素差异。Energy 用 logits 的负 log-sum-exp 评分。OutputDrift 比较参考分布与当前窗口的 Jensen-Shannon 距离。",
    params: [
      ["temperature", "1", "Energy 温度"],
      ["window_size", "256", "输出分布监测窗口长度"],
      ["false_positive_rate", "0.05", "目标误报率"],
    ],
    code: "from aegis_lab.anomaly import Energy, OutputDrift\n\nenergy = Energy(model).calibrate(clean_loader)\nflags = energy.predict(images)\nmonitor = OutputDrift(window_size=128)\nmonitor.fit(reference_probabilities)\nmonitor.calibrate(calibration_probabilities)\nevents = monitor.update(new_probabilities)",
    refs: [["Energy · NeurIPS 2020", "https://arxiv.org/abs/2010.03759"]],
  },
  {
    id: "evaluation",
    name: "安全评估",
    subtitle: "统一配置与实验产物",
    items: [
      "正常 / 鲁棒准确率",
      "逐类 / 均衡准确率",
      "AUROC / FPR / TPR",
      "规避成功率 / 耗时",
      "JSON / CSV / HTML 报告",
    ],
    description:
      "对同一测试划分运行不同方法；检测阈值来自独立干净校准集。逐类记录用于比较长尾模型，导出配置、权重、案例和报告。",
    params: [
      ["sample_limit", "0", "0 表示完整测试集"],
      ["budgets_255", "[2,4,8]", "扰动预算列表"],
      ["seeds", "[0]", "实验种子"],
      [
        "stage",
        "plan",
        "plan / train / evaluate / detect / defend / anomaly / all",
      ],
    ],
    code: 'from aegis_lab.evaluation import evaluate\n\nresults = evaluate(model, test_loader, [\n    {"method": "pgd20", "epsilon_255": 8},\n    {"method": "cw_linf", "epsilon_255": 8}\n])',
    refs: [],
  },
];
export const docs = [
  {
    id: "install",
    title: "安装",
    intro: "GPU 服务器先安装与 CUDA 对应的 PyTorch，再在项目目录安装算法库。",
    code: 'python -m pip install -e ".[attacks]"\n\n# 可选工作台\npython -m pip install -e ".[attacks,workbench]"\n\naegis --help',
    notes: [
      "Python 3.10+；安装名 aegis-lab，导入名 aegis_lab。",
      "wheel 安装：python -m pip install ./packages/aegis_lab-0.4.0-py3-none-any.whl",
    ],
  },
  {
    id: "quickstart",
    title: "快速开始",
    intro: "包装自己的 PyTorch 模型，让归一化和模型处于同一条可微计算路径。",
    code: 'import torch\nfrom aegis_lab import Classifier, create_model, attack\n\nnet = create_model("cifar_resnet18", num_classes=10)\ncheckpoint = torch.load("best_weights.pt", weights_only=True)\nnet.load_state_dict(checkpoint["state_dict"])\nmodel = Classifier(net,\n    mean=(0.4914, 0.4822, 0.4465),\n    std=(0.247, 0.2435, 0.2616)).cuda().eval()\n\nimages, labels = next(iter(test_loader))\nimages, labels = images.cuda(), labels.cuda()\nx_adv = attack(model, images, labels, "pgd20",\n    epsilon_255=8, step_size_255=2)\npredictions = model(x_adv).argmax(1)',
    notes: [
      "test_loader 返回 [0,1] 的 NCHW 图像和整数类别标签。",
      "自定义模型输出 [N,C] logits；把 net 换成自己的 torch.nn.Module。",
    ],
  },
  ...[
    "attacks",
    "detection",
    "training",
    "preprocessing",
    "ensembles",
    "evaluation",
  ].map((id) => {
    const m = methods.find((x) => x.id === id);
    return {
      id,
      title: m.name + "教程",
      intro: m.description,
      code: m.code,
      notes:
        id === "detection"
          ? [
              "拟合使用训练子集，calibrate 使用独立干净验证子集。",
              'load_detector("detector.pt", model) 恢复检测器。',
              "异常检测接口见“方法与功能 → 异常检测”。",
            ]
          : id === "training"
            ? [
                "class_counts 使用实际训练划分的类别样本数。",
                "命令行：aegis train --config configs/cifar10-at-bsl.json",
                "普通蒸馏传入 ERM 教师；防御蒸馏先用 distill_teacher 训练高温教师。",
              ]
            : [],
    };
  }),
  {
    id: "api",
    title: "API 索引",
    intro: "算法接口可直接用于自己的 Python 项目。",
    code: "aegis_lab.Classifier(model, mean, std)\naegis_lab.create_model(architecture, num_classes)\naegis_lab.attack(model, images, labels, method, **options)\naegis_lab.training.train_epoch(...)\naegis_lab.detection.{Confidence, Entropy, Consistency,\n    FeatureSqueezing, Mahalanobis, LayerRegression}\naegis_lab.detection.load_detector(path, model)\naegis_lab.preprocessing.preprocess(images, transforms)\naegis_lab.ensembles.Ensemble(models, method, transforms)\naegis_lab.anomaly.{InputStatistics, Energy, OutputDrift}\naegis_lab.evaluation.{evaluate, evaluate_detector,\n    classification_metrics, detection_metrics, adaptive_attack}",
    notes: [
      "fit(loader) → self；score(images) → 每张图的分数。",
      "calibrate(loader, false_positive_rate) → self；predict(images) → 告警张量。",
      "save(path) 保存检测参数和阈值；各方法参数见“方法与功能”。",
    ],
  },
  {
    id: "reproduce",
    title: "完整复现",
    intro: "统一配置管理训练、评估和网站成果导出。",
    code: "aegis experiment --stage plan\naegis experiment --stage train --group main --download\naegis experiment --stage train --group long_tail --download\naegis experiment --stage evaluate\naegis experiment --stage detect --group main\naegis experiment --stage defend --group main\naegis experiment --stage anomaly --group main\n\naegis export --include-weights\nnpm run build",
    notes: [
      "配置：configs/experiment-plan.json；42 个训练任务，16 组攻击参数。",
      "--only cifar10-cifar_resnet18-erm-ir1-s0 只运行匹配任务。",
      "--smoke 在独立目录进行两轮小样本流程检查。",
      "服务器操作见 DSH_SERVER_HANDOFF.md。",
    ],
  },
  {
    id: "workbench",
    title: "可选工作台",
    intro: "使用图形界面配置模型、数据、训练与评估任务。",
    code: 'python -m pip install -e ".[workbench,attacks]"\npython -m uvicorn backend.app:app --host 127.0.0.1 --port 8765\n\n# 另一个终端\nnpm install\nnpm run dev',
    notes: [
      "工作台使用本机 /api 服务；公开成果页读取导出的 data/results.json。",
    ],
  },
];
