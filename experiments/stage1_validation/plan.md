# 第一阶段全面验收计划

更新时间：2026-03-25

## 目标

围绕老师交给我们的第一阶段任务，验证当前 Nexus 多模态 embedding 代码栈是否已经做到：

- 真实可用
- 功能闭环
- 能支撑后续第二阶段训练
- 在共享机器环境中可安全执行

这里的“第一阶段”具体指：

- 多模态微调
- 多模态推理
- 本地多模态评测
- MMEB 相关训练 / 评测数据准备
- 隔离环境与共享 GPU 安全工具
- 训练输出到推理 / 评测的顺畅衔接

本轮验收不满足于“单元测试通过”，而是要做真实运行时验证。

## 验收原则

- 只使用隔离环境 `costa`，不污染本地 `base` 环境。
- 只使用空闲 GPU，绝不影响别人已经在跑的任务。
- 尽量优先使用本地现成资产做真实验证：
  - 本地 backbone：`/tmp/qwen2vl2b_local`
  - 本地 LoRA smoke 输出：`/tmp/nexus_mm_smoke_train`
  - 仓库自带示例数据：`examples/multimodal_retrieval/`
  - 已经下载过的真实训练子集：`/tmp/public_mmeb_raw`
- 发现问题先修复，再重新执行相关验收项。
- 计划、过程记录、结果摘要统一收敛到本文件夹。

## 验收矩阵

### A. 回归与基础链路

执行：

- `pytest tests/multimodal_retrieval -q`
- `bash tools/multimodal_retrieval/validate_stack.sh`

目的：

- 确认多模态工具、CLI、数据脚本、dry-run 准备链路在当前最新代码下仍然通过

### B. GPU 安全工具

执行：

- `tools/multimodal_retrieval/check_idle_gpus.py` 直接探测模式
- `tools/multimodal_retrieval/check_idle_gpus.py --input <gpu快照>` 文件输入模式

目的：

- 确认共享机器上选择空闲 GPU 的辅助工具真正可用
- 同时覆盖“可直接探测”和“受限环境使用快照文件”的两种情况

### C. 运行时推理与评测闭环

使用：

- 基础模型：`/tmp/qwen2vl2b_local`
- 训练输出：`/tmp/nexus_mm_smoke_train`
- 示例评测集：`examples/multimodal_retrieval/data/eval`

执行：

- 单卡真实推理
- 多设备推理一致性校验
- 基础模型的本地真实评测
- LoRA 训练输出目录的本地真实评测

目的：

- 确认推理不是“只能 import”
- 确认多设备路径真实可用
- 确认评测链路真实可用
- 确认训练产物能直接被后续推理 / 评测加载

### D. 配置文件模式训练入口

执行：

- 使用 split-json 配置文件模式
- 基于本地 backbone 和仓库示例训练数据做 one-step LoRA smoke

目的：

- 确认真实训练时，不只是命令行参数模式能跑，配置文件模式也能真正跑通

### E. 真实训练数据准备链路

使用：

- 本地已经收集的真实子集：`/tmp/public_mmeb_raw`

执行：

- 重新运行 `prepare_mmeb_v2_train_data.py`
- 为新的验收输出目录重新生成 stage 配置

目的：

- 确认 manifest 驱动的本地真实训练数据准备链路可复现

### F. MMEB v2 评测准备链路

执行：

- `prepare_mmeb_v2_eval_data.py`

覆盖：

- image 代表数据集
- video 代表数据集
- visdoc 代表数据集

目的：

- 确认评测准备脚本仍能正确生成转换命令与评测配置

## 当前重点风险点

- LoRA 输出目录如果不能直接被加载，会破坏训练到评测的闭环
- 多设备推理如果只是在接口上支持、实际不工作，会影响后续大规模编码 / 评测
- 配置文件模式可能解析正确，但真实运行时仍然失败
- GPU 空闲检测工具在不同 `nvidia-smi` 输出格式下可能存在兼容性问题
- MMEB eval prep 仍然依赖本地路径与远端路径之间的正确解析

## 退出标准

- 回归测试全部通过
- 基础模型运行时验证成功
- LoRA 输出目录运行时验证成功
- 配置文件训练 smoke 成功
- 真实本地 public train 子集转换成功
- eval prep 对 image / video / visdoc 代表数据集的 dry-run 成功
- 所有结果写入 `results.md`
