# 第一阶段全面验收结果

更新时间：2026-03-25

本文件记录第一阶段代码栈的真实验收结果。这里只写已经确认的事实，不写主观推测。

## 当前状态

本轮计划内的主要验收块已完成，且关键闭环已经跑通：

- 回归测试通过
- 运行时推理闭环通过
- 训练输出到评测闭环通过
- 配置文件模式训练与评测通过
- 真实 public train 子集准备通过
- MMEB eval prep 的 image / video / visdoc dry-run 通过

剩余工作主要是：

- 把结果继续沉淀进项目进度文档
- 把环境与 backbone 选择说明单独沉淀
- 做 git 提交
- 后续如有需要，再扩大真实数据和真实训练规模

## 已确认问题与修复

### 2026-03-25：发现并修复 LoRA 训练输出无法直接加载的问题

现象：

- `/tmp/nexus_mm_smoke_train` 这样的 LoRA / PEFT 输出目录，之前不能直接被 `MultimodalEmbedder` 加载。

影响：

- 训练完成后，不能直接把训练输出目录拿去做推理或评测。
- 这会破坏第一阶段最关键的闭环之一：
  - 训练
  - 保存输出
  - 加载输出
  - 做评测

根因：

- 运行时加载逻辑默认假设 `model_name_or_path` 一定是一个普通 Hugging Face 模型目录，并且里面有 `config.json`。
- 但 LoRA / PEFT 输出目录通常只有：
  - `adapter_config.json`
  - adapter 权重
  - tokenizer / processor 文件
- 因此直接走 `AutoConfig.from_pretrained()` 会失败。

已做修复：

- 在 `Nexus/modules/multimodal.py` 中补上 PEFT-aware 的 backbone 加载逻辑。
- 对 adapter 目录增加 processor 回退逻辑。
- 在训练侧保留 adapter 目录继续作为可训练模型路径的能力。
- 在 `Nexus/training/embedder/multimodal_retrieval/modeling.py` 中避免对已经是 PEFT 模型的对象再次做一层 LoRA 包裹。
- 补充了 adapter 目录加载的回归测试。

修复后立即复核结果：

- `pytest tests/multimodal_retrieval/test_multimodal_utils.py -q` 通过，结果为 `8 passed`
- 重新加载 `/tmp/nexus_mm_smoke_train` 成功，模型对象类型为 `PeftModel`

### 2026-03-25：发现 GPU 空闲检测工具在当前环境下对 `nounits` 格式不兼容

现象：

- `check_idle_gpus.py` 默认调用的 `nvidia-smi --format=csv,noheader,nounits` 在当前环境中会触发 NVML 错误。
- 但带单位格式 `--format=csv,noheader` 是可用的。

影响：

- 这会导致 GPU helper 在一些受限环境里直接失败，影响共享机器上的可用性。

处理策略：

- 已决定修复为：
  - 先尝试 `nounits`
  - 失败后自动回退到带单位格式

状态：

- 解析与回退逻辑已修复，并补了测试。
- `--input` 快照模式已真实验证成功。
- 当前 Python 子进程内的 live 模式仍然受到 NVML 环境限制影响，这是运行环境层面的限制，而不是脚本解析逻辑本身错误。
- 在共享 GPU 场景下，当前可稳定使用的方案是：
  - 先通过 shell 获取 GPU 快照
  - 再用 `--input` 模式计算空闲卡

## 已完成的验收项

### 环境补齐

已完成：

- 在隔离环境 `costa` 中安装：
  - `faiss-cpu`
  - `pytrec_eval`

说明：

- 标准依赖路径已经补齐。
- 代码侧仍保留了合理的鲁棒性处理：
  - 依赖存在时走标准路径
  - 缺依赖时不让整个代码栈在 import 阶段直接崩溃

### 回归测试

已完成：

- `pytest tests/multimodal_retrieval -q`

结果：

- `31 passed`

### 验证脚本

已完成：

- `bash tools/multimodal_retrieval/validate_stack.sh`

结果：

- 全链路通过
- 包括：
  - `py_compile`
  - `pytest`
  - inventory / manifest 导出
  - train / eval conversion smoke
  - eval prep dry-run

### GPU 安全工具

已完成：

- `check_idle_gpus.py --input /tmp/nexus_stage1_validation_gpu_snapshot.csv`

结果：

- 成功识别当前忙卡为 `GPU 3`
- 输出空闲卡为：
  - `0,1,2,4,5,6,7`

### 运行时推理与评测闭环

已完成：

- 基础模型单卡推理
- 基础模型多设备一致性验证
- 基础模型本地评测
- LoRA 输出目录单卡推理
- LoRA 输出目录本地评测

使用资产：

- 基础模型：`/tmp/qwen2vl2b_local`
- LoRA 输出：`/tmp/nexus_mm_smoke_train`
- 运行时报告：`experiments/stage1_validation/runtime_outputs/runtime_validation_report.json`

结果摘要：

- 基础模型 query / passage embedding 维度均为 `1536`
- 多设备一致性：
  - `query_allclose = true`
  - `passage_allclose = true`
  - `max_query_abs_diff = 0.0`
  - `max_passage_abs_diff = 0.0`
- 基础模型 toy eval：
  - `ndcg_at_10 = 1.0`
  - `recall_at_10 = 1.0`
- LoRA 输出目录已经可以直接被加载和评测
- LoRA 输出 toy eval：
  - `ndcg_at_10 = 1.0`
  - `recall_at_10 = 1.0`

对应文件：

- `experiments/stage1_validation/runtime_outputs/base_eval/summary.md`
- `experiments/stage1_validation/runtime_outputs/adapter_eval/summary.md`
- `experiments/stage1_validation/runtime_outputs/runtime_validation_report.json`

### 配置文件模式训练

已完成：

- `python -m Nexus.training.embedder.multimodal_retrieval --model_config ... --data_config ... --training_config ...`

使用配置：

- `experiments/stage1_validation/configs/train_model_config.local.json`
- `experiments/stage1_validation/configs/train_data_config.local.json`
- `experiments/stage1_validation/configs/train_training_config.local.json`

结果：

- 使用空闲 `GPU 6`
- one-step LoRA smoke 成功完成
- 输出目录：`/tmp/nexus_stage1_config_train`
- `train_loss = 0.57421875`

对应摘要文件：

- `experiments/stage1_validation/config_run_summaries/config_train_smoke_result.json`

### 配置文件模式评测

已完成：

- `python -m Nexus.evaluation.multimodal_retrieval --eval_config ... --model_config ...`

使用配置：

- `experiments/stage1_validation/configs/eval_config.local.json`
- `experiments/stage1_validation/configs/eval_model_adapter.local.json`

结果：

- 成功加载新的配置化训练输出 `/tmp/nexus_stage1_config_train`
- toy eval 结果：
  - `ndcg_at_10 = 1.0`
  - `recall_at_10 = 1.0`

对应摘要文件：

- `experiments/stage1_validation/config_run_summaries/config_eval_adapter_summary.md`
- `experiments/stage1_validation/config_run_summaries/config_eval_adapter_eval_results.json`

### 真实 public train 子集准备

已完成：

- 重新运行 `prepare_mmeb_v2_train_data.py`

输出位置：

- `experiments/stage1_validation/data_preparation/train_public_subset/`
- `experiments/stage1_validation/data_preparation/train_public_subset_configs/`

结果：

- 成功写出转换摘要 `conversion_summary.json`
- 当前结果显示：
  - `image/HatefulMemes.jsonl`
  - `visdoc/HatefulMemes.jsonl`
- 生成了新的：
  - `stage_a_data_config.json`
  - `stage_b_data_config.json`
  - `stage_c_data_config.json`

### MMEB eval prep dry-run

已完成：

- 重新运行 `prepare_mmeb_v2_eval_data.py --dry-run`

覆盖数据集：

- `HatefulMemes`
- `MSVD`
- `ViDoRe_arxivqa`

输出位置：

- `experiments/stage1_validation/data_preparation/eval_prep/`

结果：

- dry-run 成功
- 已写出：
  - `dry_run_report.json`
  - `HatefulMemes.eval_config.json`
  - `MSVD.eval_config.json`
  - `ViDoRe_arxivqa.eval_config.json`

## 需要继续留意的观察项

### 观察项一：`HatefulMemes` 在当前 manifest 中同时出现在 `image` 和 `visdoc`

现象：

- 在 `prepare_mmeb_v2_train_data.py --stage stage_a --sources HatefulMemes` 这次真实复跑里，工具同时生成了：
  - `image/HatefulMemes.jsonl`
  - `visdoc/HatefulMemes.jsonl`

直接原因：

- 当前 `MMEB_v2_manifest.json` 中，`HatefulMemes` 同时存在于 `train.image` 和 `train.visdoc`

判断：

- 这不是代码崩溃问题，脚本行为和 manifest 是一致的。
- 但从数据语义上看，这个分配是否完全合理，建议和数据收集负责人后续一起复核。

### 观察项二：当前真实闭环验证的 backbone 是 `Qwen2-VL-2B-Instruct`

说明：

- 这次第一阶段验收里，真正完成训练 / 推理 / 评测闭环验证的 backbone 是本地离线 `Qwen2-VL-2B-Instruct`
- 代码层已经兼容 `Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Llava-Next`
- 但除 `Qwen2-VL-2B-Instruct` 外，其他 backbone 目前仍属于“代码支持已到位、真实大闭环尚未逐个复验”的状态

对应说明文档：

- `experiments/stage1_validation/environment_and_backbone.md`

## 结论

截至目前，可以给老师一个比较扎实的结论：

- 第一阶段里我负责的 `codebase` 整理部分已经不仅“能跑”，而且经过了真实训练、真实推理、真实评测、真实数据准备和多设备一致性检查。
- 核心闭环已经成立：
  - 训练
  - 保存输出
  - 重新加载输出
  - 本地评测
- 相关过程文件、配置文件和结果摘要已经收敛到：
  - `experiments/stage1_validation/`
- 环境与 backbone 的边界、已验证范围、尚待老师明确的信息，已经单独写入：
  - `experiments/stage1_validation/environment_and_backbone.md`
