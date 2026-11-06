# Result: proxy-model-commit-cleanup

- Status: Done
- Owner: guanzhang

## Summary

本地完成新workspace的功能归组、逐提交脱敏、同功能文档合并，以及PP Timer逐iter本地落盘。
计算器只保留两个proxy预设。单机8卡DDP/HFSDPv2验证均完成40步。
训练配置和集群脚本未纳入功能提交；没有push、PR或合入上游。

## Verification

被测运行时代码为新分支 `708e44495`；后续改动仅计算器和文档。相关Python单测和pre-commit通过，
计算器19项Node检查及浏览器检查通过。未运行上游CI或新的16机验证。

| 验收项 | 实际结果 | 判定 |
| --- | --- | --- |
| 功能归属（V1） | 25条来源拆分到7个本地功能、5个PR及后置实验内容 | 通过 |
| 逐提交脱敏（V2） | 每条新增commit的消息、路径和树检查；生产preset及派生断言移除 | 通过 |
| 计算器（V3） | 两个独立预设；Node19项与浏览器切换/导入导出检查通过 | 通过 |
| 组合行为（V4） | DDP/HFSDPv2均40步exit0、有限loss/grad、0NaN/跳步 | 通过 |
| Timer（V4） | 每组320GPU+320CPU文件，普通开关关闭及窗口冲突时仍覆盖40步；0无效端点 | 通过 |
| Source保护（V5） | 原分支和5个未提交文件保留；新分支独立历史 | 通过 |

实际数据、环境变量及验证限制见[统一验证说明](../../developer/proxy_validation.md)。
各功能合并文档见[索引](../../developer/proxy_model_changes.md)。
