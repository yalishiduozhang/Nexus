# 第一阶段全面验收计划

更新时间：2026-03-25

## 一、验收目标

围绕老师交给我们的第一阶段任务，确认当前 Nexus 多模态 embedding 代码栈是否已经做到：

- 真实可用
- 功能闭环
- 可复现
- 不依赖污染 `base` 环境
- 在共享 GPU 机器上可安全执行

这里的“第一阶段”具体指：

- 多模态微调
- 多模态推理
- 本地多模态评测
- MMEB v2 训练 / 评测数据准备
- 隔离环境与共享 GPU 工具
- backbone 适配与加载可用性

## 二、本轮重点关注的问题

本轮后半段重点补的是下面几类真实问题：

1. `create_conda_env.sh` 路径不鲁棒
2. `validate_stack.sh` 依赖机器私有 `VLM2Vec` 路径
3. 只有 toy eval / dry-run，没有真实 MMEB 子集 end-to-end eval
4. 评测配置默认把缓存写到仓库根目录
5. 重新生成 eval 配置时会被网络环境绑死
6. eval 覆盖写入时失败会先删旧数据
7. backbone family 缺少成体系的真实验证产物

## 三、验收原则

- 只使用隔离环境，不污染本地 `base`
- 使用 GPU 前先检查空闲情况
- 不占用别人已经在跑任务的 GPU
- 对外汇报时只写真实发生并确认过的结果
- 过程性文件、配置和结果尽量收敛到 `experiments/stage1_validation/`

## 四、验收矩阵与当前状态

### A. 回归与基础链路

状态：已完成

执行：

- `pytest tests/multimodal_retrieval -q`
- `bash tools/multimodal_retrieval/validate_stack.sh`

当前结果：

- `39 passed`
- `validate_stack.sh` 全链路通过

### B. GPU 安全工具

状态：已完成

执行：

- `check_idle_gpus.py`
- `check_idle_gpus.py --input <gpu快照>`

目的：

- 确认共享机器上选择空闲 GPU 的辅助工具可用

### C. 运行时推理与本地评测闭环

状态：已完成

使用：

- 基础模型：`/tmp/qwen2vl2b_local`
- LoRA 输出：`/tmp/nexus_mm_smoke_train`

已完成：

- 基础模型单卡推理
- 多设备一致性验证
- 基础模型本地 toy eval
- LoRA 输出重载
- LoRA 输出本地 toy eval

### D. 配置文件模式训练入口

状态：已完成

已完成：

- one-step LoRA smoke 训练
- 配置化评测

### E. 真实训练数据准备链路

状态：已完成

已完成：

- `prepare_mmeb_v2_train_data.py` 真实 public train 子集转换
- stage 配置生成

### F. MMEB v2 评测准备链路

状态：已完成

已完成：

- image / video / visdoc 代表数据集 dry-run
- `ViDoRe_arxivqa` 真实转换
- `eval_config` 收敛到实验目录缓存
- 支持 `--write-configs-only`
- 覆盖写入改成 staging 安全替换

### G. 真实 MMEB 子集 end-to-end 评测

状态：已完成

已完成：

- 使用空闲 `GPU 2` 运行 `ViDoRe_arxivqa`
- 结果写入 `mmeb_real_eval/prepared_eval/_results/ViDoRe_arxivqa/summary.md`

### H. Backbone family 验证矩阵

状态：已完成

已完成：

- 当前 `costa` 环境验证：
  - `qwen2_vl`
  - `qwen2_5_vl`
  - `llava_next`
- 隔离 `transformers 4.57.3` 环境补充验证：
  - `qwen3_vl`

### I. `Llava-Next` family-level 全流程 smoke 验证

状态：已完成

已完成：

- 使用公开 tiny checkpoint `optimum-internal-testing/tiny-random-llava-next-mistral`
- base toy eval
- one-step LoRA smoke 训练
- adapter 重载与 runtime validation
- `ViDoRe_arxivqa` 真实 MMEB 子集评测
- 修复真实运行暴露出的 `image_sizes` batch 保留问题

## 五、当前结论

第一阶段验收已经达到退出标准。

更具体地说，当前已经满足：

- 训练、推理、评测链路可用
- MMEB 相关数据准备工具可用
- 至少一个真实 backbone 完成完整闭环
- 所有声明支持的 backbone family 完成真实加载验证
- `Llava-Next` family 也已完成真实工程闭环 smoke
- 环境和路径问题有明确解决方案

## 六、建议的下一步

如果后续进入第二阶段，建议按下面顺序推进：

1. 与老师确认最终 backbone family 与参数规模
2. 建立正式训练环境，优先满足 `transformers>=4.57.3`
3. 与负责数据收集的同学对齐 train mixture 与模态归类
4. 在当前 codebase 上准备正式训练 recipe 和 MMEB 全量评测计划
