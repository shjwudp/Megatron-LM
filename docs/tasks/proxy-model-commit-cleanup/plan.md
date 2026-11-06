# Plan: proxy-model-commit-cleanup

- Status: Draft
- Owner: guanzhang

## Implementation

- [x] 核对来源与5个工作区文件，创建独立worktree。
- [x] 按7个本地功能和5个上游PR抽取代码，同功能文档合并。
- [x] 保留上游PR作者、来源和独立提交边界。
- [x] 逐提交名称/路径与生产配置脱敏，源分支保持不变。
- [x] 合并计算器的公式和交互改动，只保留两个独立proxy预设。
- [x] 增加PP Timer dump-only模式，并验证与旧采样行为兼容。
- [x] 运行相关单元测试、计算器Node/浏览器测试与pre-commit。
- [x] 单机8卡DDP和HFSDPv2各40步，验收逐iter文件、端点、snapshot和nsys。

## Delivery

本轮只提交功能、通用单测、工具及合并文档。训练配置和任务脚本仍在仓库外验证目录，
没有创建新的16机任务，没有push或创建PR。源代码分组和测试结论见Result及developer索引。
