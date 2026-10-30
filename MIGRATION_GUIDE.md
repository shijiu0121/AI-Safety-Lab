# 0.4.0 服务器迁移

1. 将工程复制到服务器，保留 aegis_lab、backend、configs、scripts、docs、src、public 与项目配置文件。
2. 安装 CUDA 版 PyTorch，在项目目录执行 `python -m pip install -e ".[attacks]"`。
3. 按 [DSH_SERVER_HANDOFF.md](DSH_SERVER_HANDOFF.md) 执行小样本检查和正式实验。
4. 正式计划为 `configs/experiment-plan.json`，查看任务：`aegis experiment --stage plan`。
5. 运行结束执行 `aegis export --include-weights`，把结果导出到 public/data。
6. 网站构建：`npm install`、`npm run build`；dist 为静态成果网站。

本机 .venv、node_modules、.package-cache 不需要复制。训练结果保存在 data/experiments-v04。Python 库的 wheel 位于 packages。

原 backend.server_plan 与 configs/server-plan.json 保留为旧五任务工作流；新实验使用 aegis experiment 和本页链接的服务器说明。
