# 多模态 Embedding 项目总计划

更新时间：2026-03-25

## 一、项目总目标

基于 `Nexus` 整理出一个可真实用于多模态 embedding 研发的工程底座，覆盖：

- 多模态微调
- 多模态推理
- 多模态评测
- MMEB v2 相关数据准备
- 隔离环境与共享 GPU 安全执行

在此基础上，为第二阶段正式训练做好准备，最终目标是在 `MMEB v2` 上尽量逼近并争取超过 `Qwen3-VL-Embedding-8B` 这一公开强基线。

## 二、当前阶段定位

当前工作的主目标已经明确收敛到老师布置的第一阶段，重点是把下面两件事做扎实：

- 把 `Nexus` 整理成可真实使用的多模态 embedding codebase
- 为后续第二阶段留下真实可复现、低故障率的训练 / 推理 / 评测底座

本轮不把重心放在“直接冲榜”，而是先确保第一阶段做得真实、可验收、可交接。

## 三、当前已经完成的核心工作

### 3.1 多模态主链路

已经在 `Nexus` 中补齐：

- `training/embedder/multimodal_retrieval/`
- `inference/embedder/multimodal_retrieval/`
- `evaluation/multimodal_retrieval/`
- `examples/multimodal_retrieval/`

也就是说，训练、推理、评测三条主链路已经在仓库内形成了明确入口，而不是停留在零散模块级别。

### 3.2 面向 MMEB v2 的能力补齐

已经补齐并验证的能力包括：

- 图片 / 视频 / 视觉文档输入支持
- 多模态字段标准化
- `Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Llava-Next` 代码层兼容
- `media_root / image_root / video_root` 显式路径管理
- 多设备推理
- LoRA 训练产物重新加载

### 3.3 数据工具链

已经补齐：

- MMEB v2 inventory 文档
- MMEB v2 machine-readable manifest
- train 数据转换脚本
- eval 数据转换脚本
- public data 下载 / 规划工具
- MMEB v2 train/eval 准备脚本

这意味着第一阶段不仅整理了模型代码，也把后续实际训练所需的数据工具链往前推进了。

### 3.4 环境与执行安全

已经补齐：

- 隔离环境构建脚本
- 环境配置文件
- 共享 GPU 空闲检测工具
- 一键验证脚本

并且当前验收要求始终保持：

- 不污染本地 `base` 环境
- 不占用其他人正在跑任务的 GPU

### 3.5 第一阶段真实验收

已经单独建立：

- `experiments/stage1_validation/`

其中集中沉淀了：

- 中文计划
- 中文结果
- 运行时验证报告
- 配置化训练 / 评测 smoke 配置
- 真实数据准备结果
- backbone 与环境说明

## 四、当前已确认的真实结论

截至目前，下面这些事实已经被真实验证，而不是停留在理论设计上：

- `pytest tests/multimodal_retrieval -q` 通过，结果为 `31 passed`
- `tools/multimodal_retrieval/validate_stack.sh` 通过
- 本地离线 `Qwen2-VL-2B-Instruct` 已完成真实 smoke 验证
- LoRA 训练输出目录已经可以直接用于后续推理和评测
- 配置文件模式训练与评测可以跑通
- 真实 public train 子集准备可以复现
- MMEB v2 的 image / video / visdoc 代表评测集 dry-run 可以复现

## 五、当前仍在进行中的事项

虽然第一阶段主体已经成型，但下面这些工作仍处于“继续推进”状态：

- 等待或推动更大范围的真实 public train mixture 准备
- 等待老师或项目组进一步明确第二阶段最终 backbone 选择
- 基于最终 backbone 和数据规模，制定更正式的训练配方
- 为后续 MMEB v2 子集评测和正式 leaderboard 评测准备统一配置

## 六、当前尚未启动或尚未完成的事项

下面这些属于第二阶段及以后，不应在当前阶段夸大为“已经完成”：

- 大规模正式训练
- 完整 MMEB v2 leaderboard 提交
- 与 `Qwen3-VL-Embedding-8B` 的正式分数对比
- 迭代数据混合比例、instruction、pooling 和负样本策略后的系统 ablation

## 七、里程碑拆分

### M0：多模态骨架接入 Nexus

状态：已完成

范围：

- 多模态训练 / 推理 / 评测主链路
- 示例配置与示例脚本
- 首个本地里程碑提交

### M1：面向 MMEB v2 的能力加固

状态：已完成

范围：

- 视频输入支持
- Qwen 系列兼容
- 媒体根路径管理
- 多设备推理修复
- 文档与跟踪记录

### M2：数据工具与公共数据准备

状态：已完成

范围：

- inventory / manifest
- train / eval 数据转换
- public data 规划与局部真实 smoke

### M3：第一阶段全面验收

状态：已完成

范围：

- 回归测试复跑
- 运行时推理 / 评测闭环验证
- 配置文件模式训练与评测
- 真实数据准备复跑
- 环境与 backbone 说明沉淀

### M4：第二阶段训练方案准备

状态：进行中

范围：

- 确认最终 backbone family 与参数规模
- 确认数据混合与阶段划分
- 准备正式训练 recipe

### M5：正式训练与 MMEB v2 冲榜

状态：未开始

范围：

- 大规模训练
- MMEB v2 子集与全量评测
- 结果分析与迭代

## 八、当前最优先的下一步

1. 完成本轮第一阶段改动的 git 提交，形成可追踪里程碑。
2. 与老师确认第二阶段最终更偏向哪条 backbone 路线，以及大致算力预算。
3. 与负责数据收集的同学对齐 manifest 中个别数据集的模态归类问题。
4. 在第一阶段代码底座不变的前提下，准备下一轮更正式的训练配方和数据混合配置。

## 九、关键风险与依赖

- 第二阶段最终 backbone 未唯一确定，会影响环境版本、显存需求和正式训练 recipe。
- MMEB v2 涉及 image / video / visdoc，多模态数据管理复杂，路径和格式必须继续保持可配置。
- 共享机器上的 GPU 使用必须持续遵守空闲检查原则。
- 大规模 public data 收集会受到磁盘空间和网络条件限制。
- 最终 leaderboard 结果依赖第二阶段正式训练与评测，不应提前做结论。

## 十、工作约束

- 不在本地 `base` 环境安装依赖。
- 所有阶段性进展都通过 `git` 记录，不随意回退用户原有改动。
- 使用 GPU 前先检查空闲情况，避免影响其他人的任务。
- 验证、计划和结果尽量集中写入中文文档，方便老师验收与汇报。
