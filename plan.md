# 多模态 Embedding 项目总计划

更新时间：2026-03-25

## 一、项目总目标

基于 `Nexus` 整理出一个可真实用于多模态 embedding 研发的工程底座，覆盖：

- 多模态微调
- 多模态推理
- 多模态评测
- MMEB v2 训练 / 评测数据准备
- 隔离环境与共享 GPU 安全执行

在这个底座之上，后续再进入第二阶段正式训练，目标是尽量逼近并争取超过 `Qwen3-VL-Embedding-8B`。

## 二、当前阶段结论

当前阶段的重心已经明确收敛到老师布置的第一阶段，即：

- 把 `Nexus` 整理成可真实使用的多模态 embedding codebase
- 让后续第二阶段可以直接在此基础上继续训练和评测

截至本轮结束，第一阶段已经达到可汇报、可交接状态。

## 三、当前已经完成的核心工作

### 1. 多模态主链路

已经在 `Nexus` 中补齐：

- `training/embedder/multimodal_retrieval/`
- `inference/embedder/multimodal_retrieval/`
- `evaluation/multimodal_retrieval/`
- `examples/multimodal_retrieval/`

### 2. 面向 MMEB v2 的工具链

已经补齐：

- MMEB v2 inventory / manifest
- train 数据转换脚本
- eval 数据转换脚本
- public data 规划 / 下载工具
- MMEB v2 train / eval 准备脚本

### 3. 环境与共享 GPU 安全工具

已经补齐：

- 隔离环境构建脚本
- 环境配置文件
- 共享 GPU 空闲检测工具
- 一键验证脚本
- backbone family 验证矩阵脚本

### 4. 第一阶段验收资产

已经集中沉淀到：

- `experiments/stage1_validation/`

其中包含：

- 中文计划
- 中文结果
- runtime 验证报告
- 详细实验分析文档
- 配置化训练 / 评测 smoke 配置
- 真实 MMEB 子集评测结果
- `qwen3_vl` 闭环实验结果
- backbone 与环境说明

## 四、当前已经真实确认的事实

下面这些已经经过真实执行验证：

- `pytest tests/multimodal_retrieval -q` 通过，结果为 `37 passed`
- `tools/multimodal_retrieval/validate_stack.sh` 通过
- `Qwen2-VL-2B-Instruct` 已完成真实训练 / 推理 / 本地评测 / 真实 MMEB 子集评测闭环
- `Qwen3-VL-2B-Instruct` 已在隔离环境中完成真实 smoke 训练 / 重载 / 本地评测 / 真实 MMEB 子集评测闭环
- LoRA 输出目录可以直接被重新加载做后续推理与评测
- 真实 public train 子集准备成功
- MMEB image / video / visdoc 评测准备工具真实可用
- `ViDoRe_arxivqa` 真实 MMEB 子集 end-to-end eval 已通过
- `Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Llava-Next` 四个 family 都完成了真实 backbone 加载验证
- `experiments/stage1_validation/experiment_analysis.md` 已形成逐实验、逐产物的详细复核记录

## 五、当前仍需诚实保留的边界

下面这些还没有完成，不应提前夸大：

- 第二阶段大规模正式训练
- 完整 MMEB v2 leaderboard 提交
- 与 `Qwen3-VL-Embedding-8B` 的正式分数对比
- `Qwen2.5-VL / Llava-Next` 的完整 benchmark 闭环

当前最准确的表述是：

- 第一阶段已经完成
- 第二阶段尚未开始正式冲榜训练

## 六、里程碑状态

### M0：多模态骨架接入 Nexus

状态：已完成

### M1：MMEB v2 能力补齐

状态：已完成

### M2：数据工具与公共数据准备

状态：已完成

### M3：第一阶段全面验收

状态：已完成

### M4：第二阶段训练方案准备

状态：待老师确认最终 backbone 后继续推进

### M5：正式训练与 MMEB v2 冲榜

状态：未开始

## 七、当前最优先的下一步

1. 提交本轮 git 版本，固化第一阶段收尾里程碑。
2. 与老师确认第二阶段最终 backbone family 和参数规模。
3. 与负责数据收集的同学对齐 train mixture 和模态归类。
4. 基于新的 backbone 选择准备正式训练 recipe。

## 八、工作约束

- 不在本地 `base` 环境安装依赖。
- 使用 GPU 前先检查空闲情况。
- 不影响其他人正在跑的 GPU 任务。
- 所有阶段性结果尽量沉淀到中文文档和 git 历史中，方便老师验收与汇报。
