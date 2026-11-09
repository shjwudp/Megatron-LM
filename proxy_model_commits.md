# Proxy model commit 对照表

更新日期：2026-11-09。上游为 [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)。
固定上游基线：`a0ef64615cc3d2574ffbaac70f324ee365f5ed34`。

本表保留前次整理的功能归组和历史来源映射，并补充最终实验使用的优化器修复。
共有 **13 条代码/功能提交：7 个归组功能、5 个独立上游 PR、1 个优化器修复**。
运行代码版本为 `4b4e7e08f678d81e13f8aca789d1897130547220`；
随后仅添加本表和文档链接，不改变运行代码。五个上游 PR 未合并成一个提交。

实验模型配置、训练入口和原始 profiling 记录作为独立压缩包分发，未纳入功能历史。
旧提交 hash 仅供追溯，不代表这些旧提交或其配置已进入本仓库的可达历史。

| # | 新 commit | 功能 / 合并来源 | 统一文档 | 实际验证 |
| --- | --- | --- | --- | --- |
| 1 | `f62fe3aac` | MFSDPv2 PP / hybrid拓扑；A + B + S的Dense-only chunk修复 | [说明](docs/developer/proxy_fsdp_parallelism.md) | Dense chunk 2项；8卡HFSDP 40步 |
| 2 | `0cf5cc058` | Muon与scalar optimizer；C + D | [说明](docs/developer/proxy_muon_optimizer.md) | 运行时代码与来源等价；8卡HFSDP 40步 |
| 3 | `ef6162d40` | Packed MTP / pipeline生命周期；G + S的MTP builder + T | [说明](docs/developer/proxy_packed_mtp.md) | GPT/schedule 78项；DDP/HFSDP各40步 |
| 4 | `d3cd55a9d` | CSA / Dense / MoE局部重算；H的重算实现 | [说明](docs/developer/proxy_selective_recompute.md) | local 12项；CP2 4项/rank；8卡组合验证 |
| 5 | `6f33f603e` | MoE确定性强制均衡；I的router控制 | [说明](docs/developer/proxy_moe_balance.md) | 2项计数/梯度检查；8卡组合验证 |
| 6 | `f7c42e9e2` | 可观测与分析工具；E + I打点 + V + W + X-NVTX；H/P通用分析工具 | [说明](docs/developer/proxy_observability.md) | EventTracker/FLOPs 77项；Timer/nsys/snapshot验收 |
| 7 | `81890d6d2` | 上游PR #8684；J；保持独立 | [说明](docs/developer/mfsdp_v2/pr_8684_muon_overlap.md) | 保留来源验证说明；本轮no-UB集成验证 |
| 8 | `28e3e8a09` | 上游PR #8782；K；保持独立 | [说明](docs/developer/mfsdp_v2/pr_8782_muon_side_stream.md) | 保留来源验证说明；本轮no-UB集成验证 |
| 9 | `fddcbb14a` | 上游PR #8856；L；保持独立 | [说明](docs/developer/mfsdp_v2/pr_8856_gradient_input_lifetime.md) | 保留来源验证说明；本轮no-UB集成验证 |
| 10 | `ebe916f84` | 上游PR #8918；M；保持独立 | [说明](docs/developer/mfsdp_v2/pr_8918_native_nccl_ub.md) | 历史UB验证；本轮未重新验证UB路径 |
| 11 | `708e44495` | 上游PR #8950；N；保持独立 | [说明](docs/developer/mfsdp_v2/pr_8950_allgather_input_registration.md) | 历史UB验证；本轮未重新验证UB路径 |
| 12 | `3568bd3ec` | 显存计算器；H计算器 + U + X-memory + 5个工作区改动 | [说明](docs/developer/proxy_memory_calculator.md) | Node 19项；浏览器检查；历史proxy snapshot校准 |
| 13 | `4b4e7e08f` | Adam/Muon 参数归属和算法对齐；修复 MFSDP 配置透传，统一 r2s10 和 head/FC1/EH 分块 | [优化器](docs/developer/proxy_muon_optimizer.md) / [验证结果](docs/tasks/proxy-optimizer-alignment/Result.md) | 42 项 GPU 检查、独立 legacy 数值对照、8 卡训练及最终两项 128 卡任务完成 |

## 历史来源到新功能的映射

下表以原索引字母区分来源；旧索引中的两个 X 明确分成 X-NVTX 和 X-memory。

| 旧编号 / commit | 新位置 |
| --- | --- |
| A `75d8ffe3f` / B `78c7974e0` | 1 |
| C `d69e8d0db` / D `e296d5ea4` | 2 |
| E `0605bac71` | 6，合并工具及buffer/precision说明 |
| F `1cffd9e16` | 训练模型配置及比较记录后置 |
| G `cb16ab5ec` | 3 |
| H `c6edccfb4` | 4重算、6 snapshot审计、12计算器 |
| I `faf2a6ae5` | 5均衡控制、6观测 |
| J `af957e6c5` / K `16f22171d` / L `75df3e669` | 7 / 8 / 9，分别独立 |
| M `8cd68ecd5` / N `eeab301ff` | 10 / 11，分别独立 |
| O `fe348e366` | 训练入口的每步内存开关后置；logger能力由6说明 |
| P `bc6821634` | 通用汇总/snapshot工具进入6；任务脚本和原始配置后置 |
| Q `721d1af20` / R `d1295fa89` | mock采样配置、集群脚本、原始实验记录后置；本轮验证复用外置冻结harness |
| S `9c28da838` | Dense-only适配进入1，MTP builder进入3；cache/任务入口后置 |
| T `0e9de176c` | 3；修复生命周期并保留GC边界验证 |
| U `4af7eb4f6` | 12；完整FSDP staging/wgrad及EP overlap估算 |
| V `189086d83` | 6；CSA FLOPs统计 |
| W `0d3212cae` / X-NVTX `b289f1ece` | 6；重算/层次NVTX及紧凑视图 |
| X-memory `89e1df611` | 12；碎片预算与snapshot比较口径 |
| 源工作区5个未提交计算器文件 | 12；三方合并交互/公式，源文件hash保持不变 |


## 最终实验与代码一致性

最终比较为 HFSDP `j-1yavtugodl` 和 DDP `j-0q1hqgfrll`。
两项任务使用同一个冻结源码包，并均完成 70 iter；实际优化器审计在全部 rank 通过。
稳定期分析范围为训练日志 iter41–70。报告及实际配置随独立的验证/报告包分发。

这次整理没有重新训练。共享分支的 2,625 个训练代码/测试常规文件和 2 个符号链接
与执行时源码包一致；Black、Pylint、isort、DCO、语法与 whitespace 检查通过。
具体数值验证、限制和命令见 [优化器任务结果](docs/tasks/proxy-optimizer-alignment/Result.md)。
原始源码包和依赖覆盖层的 SHA-256 见 [来源说明](docs/developer/proxy_experiment_release.md)。

## 验证范围

表中的 8 卡与 UB 结果属于相应功能实现时的历史验证，未声称本次重新执行。
最终两项任务使用 no-padding/no-UB 配置，因此不能把其成功当作 UB 或 symmetric-memory
路径的新验证。验证方式和限制以每一行链接的 developer 文档为准。

代码中的模型称呼统一为 proxy model。计算器仅保留先前允许的两个独立 proxy 预设。
共享历史从固定的公开上游基线重建，原始私有历史只保留上表的追溯编号。
