# 课程要求与实现对应 · 0.4.0

| 题目内容 | Python 实现 | 实验 | 展示 |
| --- | --- | --- | --- |
| 对抗样本生成 | attacks.py / _core/attacks.py | 16 组攻击参数 × 42 模型 | 预算曲线、案例、参数下载 |
| 攻击检测 | detection.py：6 方法，fit/score/calibrate/save/load | 4 模型 × 6 检测器，六攻击+自适应 PGD | ROC、AUROC、误报率、规避率 |
| 鲁棒性训练 | training.py / _core/train.py：7 模式 | 22 常规/蒸馏 + 20 长尾 | 训练曲线、逐类指标、模型卡 |
| 输入预处理 | preprocessing.py：5 变换 | PGD-AT 上逐项测试 | 方法介绍、迁移/自适应对比 |
| 模型集成 | ensembles.py：4 策略 | 3 架构 PGD-AT 模型集成 | 对比表、成员配置 |
| 异常检测 | anomaly.py：统计、Energy、OutputDrift，复用 Mahalanobis | 跨 CIFAR、3 种变换、窗口漂移 | 指标表、窗口事件、产物下载 |
| 安全评估 | evaluation.py / experiments.py / publishing.py | 同配置执行与导出 | 静态成果比较与报告 |
| 实时防护 | 原 backend/guard_api.py 本机推理与事件记录 | 后续单独确定实时系统演示 | 当前保留在可选工作台 |

库安装入口 pyproject.toml；统一配置 configs/experiment-plan.json；GPU 提示词 DSH_SERVER_HANDOFF.md。

公开成果页使用 src/PublicSite.vue，工作台位于 #/workbench/overview。训练与 GPU 性能结果在服务器完成后导出。
