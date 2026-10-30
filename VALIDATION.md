# 验证记录 · 0.4.0

日期：2026-10-30。

- 完整 CPU 测试：python -m unittest discover -s tests -v，68 项通过。
- 新增测试覆盖：攻击预算与训练状态恢复、Feature Squeezing 公式和梯度、Mahalanobis 与 NumPy 协方差计算一致性、Layer Regression 拟合下降、8 种检测/异常对象保存恢复、输出窗口累积、ROC 并列分数、集成及自适应攻击、42 任务展开、静态导出与下载路径。
- Vue/Vite 生产构建通过，28 个模块。公开页与工作台分别打包。
- 浏览器检查：首页、快速开始、实验成果导航与空数据状态。正式公开数据 results.json 当前为空。
- Python wheel 与源码包已生成，位于 packages/。
- GPU 正式训练、完整数据集评估及性能测量留待服务器运行。执行说明：DSH_SERVER_HANDOFF.md。
