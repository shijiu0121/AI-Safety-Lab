# 第三方来源与许可

本项目未将第三方攻击算法作为原创，也未把对应库的源码复制进本仓库。安装时通过依赖接入。

| 项目 | 用途 | 固定版本 | 许可与来源 |
|---|---|---|---|
| Foolbox | DeepFool-L∞/L2、模型接口设计参考 | 3.3.4 | MIT；[项目](https://github.com/bethgelab/foolbox)、[LICENSE](https://github.com/bethgelab/foolbox/blob/v3.3.4/LICENSE) |
| AutoAttack | 官方 standard 评测套件 | 提交 `a39220048b3c9f2cca9a4d3a54604793c68eca7e` | MIT；[项目](https://github.com/fra31/auto-attack)、[LICENSE](https://github.com/fra31/auto-attack/blob/a39220048b3c9f2cca9a4d3a54604793c68eca7e/LICENSE) |
| EagerPy | Foolbox 后端张量适配 | 0.30.0 | MIT；[项目](https://github.com/jonasrauber/eagerpy) |
| AT-BSL | 平衡 softmax 和对抗训练方法参考 | CVPR 2024 | [论文和代码](https://github.com/NISPLab/AT-BSL) |
| TAET | 两阶段训练与类别均衡损失方法参考 | CVPR 2025 | [论文和代码](https://github.com/Eureka-WangYuHang/TAET-Two-Stage-Adversarial-Equalization-Training-on-Long-Tailed-Distributions) |

Foolbox 作者：Jonas Rauber、Wieland Brendel、Matthias Bethge 及其贡献者。AutoAttack 作者：Francesco Croce、Matthias Hein 及其贡献者。各依赖安装包保留原有许可文件；使用与再分发应保留相应版权和许可。

本工程还使用 PyTorch/TorchVision、NumPy、Pillow、SciPy、FastAPI、Uvicorn、Pydantic、Vue、Vite 等依赖，各自遵循原项目许可。准确依赖版本同时写入实验报告，运行环境升级后应重新验证。

## 0.4.0 检测和异常方法来源

以下模块根据论文公式和官方公开实现重新编写，参数设置见 docs/API.md。

- Feature Squeezing, Xu et al., NDSS 2018: https://arxiv.org/abs/1704.01155
- Mahalanobis detector, Lee et al., NeurIPS 2018: https://github.com/pokaxpoka/deep_Mahalanobis_detector
- Layer Regression, Mumcu & Yilmaz, TMLR 2025: https://github.com/furkanmumcu/Layer-Regression (lr_modules.py: 12d8fa8fba22093875ee576e2a88fd9b8c14e76c; lr_train.py: 0a69a6d54fbfc747a46871f8848ee04eee557be3)
- Energy-based OOD Detection, Liu et al., NeurIPS 2020: https://arxiv.org/abs/2010.03759
- InputStatistics 和 OutputDrift 是本项目的统计监测实现。

Foolbox、AutoAttack、PyTorch、torchvision 与 scikit-learn 作为第三方依赖安装，保留各自许可证。
