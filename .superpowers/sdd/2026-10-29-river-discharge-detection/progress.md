# SDD ledger — plan: docs/superpowers/plans/2026-10-29-river-discharge-detection.md

Setup: git init + 初始提交（hik-sdk.py + docs/）后逐任务执行。
计划共 15 个任务。工作目录即主仓库 main 分支（新项目，无隔离需求，用户计划本身含 git init）。

- [x] Task 1: 项目脚手架与配置模块
- [ ] Task 2: 存储模块
- [ ] Task 3: 采样器
- [ ] Task 4: 水面分割模块
- [ ] Task 5: 视角配准模块
- [ ] Task 6: ROI 重建模块
- [ ] Task 7: 规则引擎
- [ ] Task 8: CNN 分类器
- [ ] Task 9: VLM 终判模块
- [ ] Task 10: 时序状态机
- [ ] Task 11: 告警接口
- [ ] Task 12: 管线编排
- [ ] Task 13: 定时调度器与入口
- [ ] Task 14: 标定工具
- [ ] Task 15: 离线回放器
Task 1: complete (commits 6ffe5f0..d09eca2, review clean)
Task 1: minor (deferred): config.py write_text/read_text 未指定 encoding=utf-8（源自简报，Windows cp936 风险）；未使用 import 残留
