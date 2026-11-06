# Design: proxy-model-commit-cleanup

- Status: Draft
- Owner: guanzhang

## Architecture

在上游基线新建提交链，先从旧提交提取功能增量，再在提交前脱敏。
不把旧敏感提交加入新分支祖先链后再另加清理commit；最终树干净不能证明中间历史干净。
现有worktree共享Git对象库，旧分支仍仅作为内部来源；重建分支不包含旧分支祖先。
本轮按此分组重建，原分支和未提交计算器文件保持不变。

已核对目标分支23个提交，HEAD `b289f1ece`；SCALE分支独有 `4af7eb4f6`、`89e1df611` 两条。
五个未提交计算器文件与SCALE对应文件均不同，按功能合并，不能覆盖式复制。
原汇总的重复X按SHA区分为X-NVTX与X-Memory。

## Changes

按下表形成12条提交：本地功能7条、上游PR5条。实际按顺序抽取和验证patch，遇到共同文件按功能hunk拆分。

| 新提交 | 功能 | 来源与拆分范围 | 主要依赖 |
| --- | --- | --- | --- |
| 1 | MFSDPv2与PP及异构HSDP拓扑集成 | A `75d8ffe3f` + B `78c7974e0` + S `9c28da838` 的纯Dense chunk校验 | 上游基线 |
| 2 | MFSDPv2 Muon+Adam混合优化器 | C `d69e8d0db` + D `e296d5ea4`，包含CSA位置偏置排除及优化器说明 | 1；D是参数分组正确性修复 |
| 3 | Packed MTP与combined 1F1B生命周期 | G `cb16ab5ec` + S的独立MTP builder修复 + T `0e9de176c` 的GC引用环修复 | 1；相应生产代码与单元测试一同迁移 |
| 4 | CSA/Dense/MoE选择性重算 | H `c6edccfb4` 的重算实现和单元测试 | 3；不含H的计算器和实验脚本 |
| 5 | MoE确定性强制均衡 | I `faf2a6ae5` 的round-robin logits、router/config及对应测试 | 独立路由功能，默认行为保持不变 |
| 6 | 训练可观测与分析工具 | E `0605bac71` + I剩余打点 + V `189086d83` + W `0d3212cae` + X-NVTX `b289f1ece`；吸收O/P中可通用化的指标说明与采集逻辑 | 1–4；集中Event Tracker、NVTX、CSA FLOPs、诊断与snapshot工具 |
| 7 | 上游PR #8684 | J `af957e6c5`，独立保留 | 2、6；现有patch上下文含profiling decorator |
| 8 | 上游PR #8782 | K `16f22171d`，独立保留 | 7 |
| 9 | 上游PR #8856 | L `75df3e669`，独立保留 | 1、3、6的本地适配 |
| 10 | 上游PR #8918 | M `8cd68ecd5`，独立保留 | FSDP集成，保持原先相对顺序 |
| 11 | 上游PR #8950 | N `eeab301ff`，独立保留 | 10 |
| 12 | Proxy显存计算器 | H的计算器 + U `4af7eb4f6` + X-Memory `89e1df611` + 五个未提交文件 | 合并不同来源的公式修正与UI改动，只保留两个proxy |

A/B共同提供HFSDP拓扑与流水线集成，建议合并。D只有参数归属修正，合入Muon功能。
S必须拆开：adapter归1、GPT builder归3，cache/训练编排后置。
T的核心修复归3，40步任务配置和集群提交参数后置。
I必须拆开：强制均衡改变logits和路由，不能被当作无副作用的观测代码。
V用于端到端TFLOPS打印，随可观测组维护，但保留独立的通用计数模块和小模型单测。
H的snapshot审计工具归6，公式/UI归12，生产重算代码归4。

可观测组会跨core、training和tools，但这是用户明确要求归并的一个功能。
已按CODEOWNERS核对：FSDP涉及megatron-fsdp，调度涉及pipeline-parallelism/gpt，
重算涉及transformer/MoE，训练日志涉及training。保持测试与代码一起，不为减少owner组拆掉回归测试。
本轮不创建PR；未来若分MR，按这些实际归属安排评审。

### 后置内容

| 来源 | 后置范围 |
| --- | --- |
| F `1cffd9e16` | 生产模型preset、训练launcher、local_run及对应实验输出 |
| O `fe348e366` | 本质是launcher开启已有日志开关；不单独建功能提交。说明归6，启动配置后置 |
| P `bc6821634` | 训练/提交入口、Canoe包装、规模实验配置；通用采集/summary工具可从依赖配置的包装中抽离到6 |
| Q `721d1af20` | 实际doc分布profile、packed mock生成及训练集成，待新的测试方案确定 |
| R `d1295fa89` | 任务提交历史及规模实验报告；只按功能保留经脱敏的验证结论 |
| G/H/I/S/T/W/X的实验部分 | launcher环境变量、规模配置、审批人、旧任务编排、cache发布与原始产物路径 |

独立MTP builder是通用代码修复，不因它位于gpt builder文件就归为待后置的模型配置。
两个计算器proxy配置是用户明确保留的例外，仅服务计算器，不自动生成或迁入训练任务脚本。

### 逐提交脱敏

统一使用proxy model；保留Muon、CSA、FSDP、FlashMLA等技术名称及现有公共API。
生产模型名称的大小写/连字符/下划线变体、路径和提交消息全部扫描。
删除生产层数、宽度、专家数、完整pipeline layout、参数量、完整argv及由此派生的对照表。
校验JSON中的config、shape、原始栈绝对路径和模型规模说明同样检查，不能只替换标题。
仅保留两个获准proxy的必要数据，移除它们与生产模型之间的映射或“同真实模型”说明。
内部任务ID、链接和原始日志留在仓库外审计记录；版本内保留通用验证方法、实际结果及限制。
旧试验通过可标为来源版本结果，新分支组合验证在实际执行后再填写。

计算器迁到 `tools/proxy_memory_calculator/`，避免依赖待后置的训练examples目录。
`local`和16机preset当前继承full，必须改为独立完整配置；schema和默认值不再借用full。
删除三个full变体、隐藏CLI入口/URL入口及对应大模型golden断言，保留重算开关本身。
默认16机proxy；EP2为另一预设。小规模形状测试验证公式，两个proxy用作集成校验。

### 上游提交保留规则

五个PR各一条，不与本地功能或其他PR合并。保留Source-PR、Upstream-Head、作者、
贡献归属及license；各自的内部适配说明在该条commit写入前完成脱敏。
重排或脱敏会改变新commit SHA；“独立保留”指来源与提交边界，不承诺原SHA不变。

### PP Timer本地落盘

本分支EventTracker原本就写JSON，没有上传路径。新增 `ENABLE_PP_TIMER_DUMP_ONLY=1`
独立开启每iter采集，覆盖普通Timer enable、range和interval限制。`finish()`同步CUDA后
在返回前写完GPU和可选CPU文件；异常step不伪造完整结果。旧采样模式在该变量关闭时不变。

每组功能保留一份合并后的developer说明，工具使用说明通过索引链接，原始实验记录留在仓库外。
两条单机验证使用与功能历史分离的固定harness；实际训练代码来自新分支。

## Verification

- **归属检查（V1，对应O1）**：25个旧SHA、5个工作区文件及新增hunk逐项映射到功能组或后置清单，禁止遗漏和重复应用。
- **历史检查（V2，对应O2）**：逐个扫描基线之后的commit消息、路径、完整tree及diff；对生产配置组合和派生数字人工复查。上游五条也逐条检查。
- **计算器检查（V3，对应O3）**：preset键仅两个；默认、CLI、UI和URL一致；Node测试通过，导入/导出与旧名称处理无隐藏生产默认值。
- **行为检查（V4，对应O4）**：按功能运行单测、导入/语法检查和pre-commit；检查依赖顺序及最终生产代码与获准来源的差异。未经运行的训练不写为通过。
- **保护检查（V5，对应O1）**：旧分支HEAD和未提交文件哈希不变，新分支从选定基线重建，原分支HEAD和dirty文件不被改写。
