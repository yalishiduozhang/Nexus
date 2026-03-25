# 第一阶段实验总分析与复核文档

更新时间：2026-03-25

本文档的目标不是简短汇总，而是把目前已经真实做过的实验尽可能完整地记录下来，方便逐项复核。每个实验都会尽量写清楚：

- 实验目的
- 执行方式
- 输入配置 / 数据 / 模型
- 输出文件
- 关键结果
- 这个实验能够证明 codebase 的哪一部分已经整理到可用状态

文档只记录已经真实执行过并有产物可查的实验；对尚未做的事情不会写成“已完成”。

---

## 一、这份文档要回答什么问题

老师交给我们的第一阶段任务，核心是：

- 把 `Nexus` 整理成支持多模态 embedding 微调、推理、评测的 codebase
- 把和 MMEB v2 对接所需的数据准备工具链整理好

如果要证明这件事“确实做得比较扎实”，不能只靠代码 diff，也不能只靠一句“测试过了”。更有说服力的证据应该是：

1. 单元测试和工具链验证通过
2. 基础模型真的能推理
3. 训练输出真的能重新加载做推理和评测
4. 配置文件模式真的能跑
5. 真实 public train 数据真的能准备出来
6. MMEB eval 数据准备工具真的能覆盖 image / video / visdoc
7. 至少一个真实 MMEB 子集真的做了 end-to-end 评测
8. 声称支持的 backbone family 真的做了真实加载验证

下面的实验就是围绕这八条展开的。

---

## 二、实验资产总览

### 2.1 主要实验目录

- `experiments/stage1_validation/`
- `examples/multimodal_retrieval/`

### 2.2 主要临时模型 / 输出目录

- 基础 backbone：
  - `/tmp/qwen2vl2b_local`
- 最早的 LoRA smoke 训练输出：
  - `/tmp/nexus_mm_smoke_train`
- 配置化训练 smoke 输出：
  - `/tmp/nexus_stage1_config_train`

### 2.3 主要隔离环境

- 第一阶段主体验收环境：
  - `/home/szn/zht/miniconda3/envs/costa`
- 为验证 `qwen3_vl` family 额外创建的隔离环境：
  - `/tmp/nexus_stage1_tf457_env`

### 2.4 主要示例数据

- 训练示例：
  - `examples/multimodal_retrieval/data/train.jsonl`
- 评测示例：
  - `examples/multimodal_retrieval/data/eval/`
- 真实 public train 子集：
  - `/tmp/public_mmeb_raw`
- 真实 MMEB visdoc 子集准备输出：
  - `experiments/stage1_validation/mmeb_real_eval/`

---

## 三、实验清单总表

| 实验编号 | 实验主题 | 核心目标 | 结果 |
| :-- | :-- | :-- | :-- |
| E1 | 多模态回归测试 | 验证新增多模态能力与工具逻辑未回归 | 通过，`37 passed` |
| E2 | 一键工具链验证 | 验证 validate_stack 全流程闭环 | 通过 |
| E3 | 基础模型运行时验证 | 验证 base model 推理、多卡一致性、本地评测 | 通过 |
| E4 | LoRA 输出运行时验证 | 验证训练输出目录可直接重载与评测 | 通过 |
| E5 | 配置文件模式训练 smoke | 验证 train CLI 的 config-file 模式可用 | 通过 |
| E6 | 配置文件模式评测 smoke | 验证 eval CLI 的 config-file 模式可用 | 通过 |
| E7 | 真实 public train 子集准备 | 验证 train 数据准备工具对真实数据可用 | 通过 |
| E8 | MMEB eval prep dry-run | 验证 image / video / visdoc 评测准备工具 | 通过 |
| E9 | 真实 MMEB 子集转换与评测 | 验证真实 MMEB 子集 end-to-end eval | 通过 |
| E10 | backbone matrix（当前环境） | 验证当前环境下 family 真实加载能力 | 通过，`qwen3_vl` 不可用属环境版本限制 |
| E11 | backbone matrix（tf 4.57.3） | 验证 `qwen3_vl` family 真实加载能力 | 通过 |
| E12 | `qwen3_vl` base toy eval | 验证 qwen3 基础评测链路 | 通过 |
| E13 | `qwen3_vl` 训练 smoke | 验证 qwen3 训练链路 | 通过 |
| E14 | `qwen3_vl` runtime validation | 验证 qwen3 base+adapter 推理与本地评测闭环 | 通过 |
| E15 | `qwen3_vl` 真实 MMEB 子集评测 | 验证 qwen3 在真实 MMEB 子集上的评测链路 | 通过 |
| E16 | example 与 config 鲁棒性修复验证 | 修复并验证示例脚本和 config-file 入口隐患 | 通过 |

---

## 四、详细实验记录

## E1. 多模态回归测试

### 1. 实验目的

验证以下部分没有因为本轮代码整理而引入回归：

- 多模态字段归一化
- image / video / page 等媒体路径解析
- train / eval 数据转换工具
- 配置文件解析
- multimodal processor 的兼容逻辑

### 2. 执行方式

执行命令：

```bash
PYTHONPATH=/home/szn/zhangx/explore/Nexus \
/home/szn/zht/miniconda3/envs/costa/bin/python -m pytest tests/multimodal_retrieval -q
```

### 3. 关键覆盖点

重点测试文件：

- `tests/multimodal_retrieval/test_multimodal_utils.py`
- `tests/multimodal_retrieval/test_data_tools.py`
- `tests/multimodal_retrieval/test_cli_config.py`

本轮额外覆盖了：

- `VLM2Vec` 根目录自动发现
- instruction-style MMEB eval row 转换
- PIL 图像序列化
- Qwen2-VL 带图截断失败后的自动重试
- train/eval 配置缓存路径写入实验目录

### 4. 输出与结果

结果：

- `37 passed`

### 5. 这个实验能证明什么

它说明：

- 代码整理不是只改了主流程，相关边界工具和辅助函数也同步被测到了
- 新补的鲁棒性修复不是“只在手工跑时偶然可用”，而是已经被单测固定下来

---

## E2. 一键工具链验证：`validate_stack.sh`

### 1. 实验目的

把“第一阶段最关键的非 GPU 依赖链路”收敛成一个可重复执行的一键入口，确认：

- Python 代码至少能编译
- 多模态测试能通过
- inventory / manifest 工具能工作
- train / eval 转换 smoke 能工作

### 2. 执行方式

执行命令：

```bash
PYTHON_BIN=/home/szn/zht/miniconda3/envs/costa/bin/python \
bash tools/multimodal_retrieval/validate_stack.sh
```

### 3. 本轮特别验证的修复点

这条验证链路专门覆盖了下面两个本轮重点修复：

- `validate_stack.sh` 不再写死私有 `VLM2Vec` 路径
- `create_conda_env.sh` 创建环境后能直接调用这条验证链路

### 4. 输出与结果

执行阶段包括：

- `[1/4] py_compile`
- `[2/4] pytest`
- `[3/4] inventory export`
- `[4/4] conversion smoke`

结果：

- 全链路通过

中间产物写到了临时目录：

- `/tmp/nexus_mmeb_validation/`

### 5. 这个实验能证明什么

它说明：

- 第一阶段整理出来的不只是几个零散脚本
- 这些工具已经能形成“安装后即可验证”的标准流程
- 对以后新建环境、交接同学、复现实验都非常重要

---

## E3. 基础模型运行时验证：base model 推理、多卡一致性、本地评测

### 1. 实验目的

验证：

- 基础多模态 backbone 真的能被 `MultimodalEmbedder` 正常加载
- 单卡推理可用
- 多设备推理结果与单卡一致
- 本地评测链路可用

### 2. 运行资产

- 模型：
  - `/tmp/qwen2vl2b_local`
- 数据：
  - `examples/multimodal_retrieval/data/media/`
  - `examples/multimodal_retrieval/data/eval/`

### 3. 执行方式

运行入口脚本：

- `tools/multimodal_retrieval/runtime_validation.py`

这次运行时使用了：

- 单卡设备：`cuda:0`
- 多卡逻辑设备：`cuda:0 cuda:1`

说明：

- 文档中的 `cuda:0` / `cuda:1` 是进程内逻辑设备编号
- 实际物理 GPU 来自空闲卡；前序记录显示该次运行是在空闲 GPU 上完成的，并没有占用正在运行别人任务的显卡

### 4. 输出文件

主报告：

- `experiments/stage1_validation/runtime_outputs/runtime_validation_report.json`

本地评测摘要：

- `experiments/stage1_validation/runtime_outputs/base_eval/summary.md`

### 5. 关键结果

来自 `runtime_validation_report.json`：

- query embedding shape：
  - `[4, 1536]`
- passage embedding shape：
  - `[4, 1536]`
- 多设备一致性：
  - `query_allclose = true`
  - `passage_allclose = true`
  - `max_query_abs_diff = 0.0`
  - `max_passage_abs_diff = 0.0`

本地 toy eval：

- `ndcg_at_10 = 1.0`
- `recall_at_10 = 1.0`

### 6. 这个实验能证明什么

它说明：

- 基础 backbone 加载是真的可用，不只是 import 成功
- 推理接口不是“只能单卡”
- 多卡聚合结果没有发生乱序或数值漂移
- 评测 runner 已经能和 embedder 打通

---

## E4. LoRA 输出运行时验证：训练产物重载与评测

### 1. 实验目的

这条实验非常关键，因为它验证的是第一阶段最容易被忽略、但实际最重要的闭环：

> 训练输出目录能不能直接作为后续推理 / 评测模型目录使用

如果这一步不通，训练和评测在工程上就是断开的。

### 2. 运行资产

- LoRA 输出目录：
  - `/tmp/nexus_mm_smoke_train`

目录中可见的关键文件包括：

- `adapter_config.json`
- `adapter_model.safetensors`
- `preprocessor_config.json`
- `tokenizer_config.json`
- `video_preprocessor_config.json`

### 3. 执行方式

仍通过：

- `tools/multimodal_retrieval/runtime_validation.py`

来完成推理与本地评测验证。

### 4. 输出文件

摘要文件：

- `experiments/stage1_validation/runtime_outputs/adapter_eval/summary.md`

总报告中对应字段：

- `adapter_model`
- `adapter_eval`

### 5. 关键结果

来自 `runtime_validation_report.json`：

- query embedding shape：
  - `[4, 1536]`
- passage embedding shape：
  - `[4, 1536]`

LoRA 输出 toy eval：

- `ndcg_at_10 = 1.0`
- `recall_at_10 = 1.0`

### 6. 这个实验能证明什么

它说明：

- 我们已经修复了“LoRA 输出目录不能直接被加载”的真实问题
- 第一阶段整理出来的是完整闭环，而不是“训练能跑、但结果不能用”

---

## E5. 配置文件模式训练 smoke

### 1. 实验目的

验证：

- 训练入口不只支持命令行传参模式
- 使用 `model_config + data_config + training_config` 的配置化方式也能真实运行

这非常重要，因为后续第二阶段正式训练不可能靠手工拼一长串命令参数维护。

### 2. 输入配置

模型配置：

- `experiments/stage1_validation/configs/train_model_config.local.json`

数据配置：

- `experiments/stage1_validation/configs/train_data_config.local.json`

训练配置：

- `experiments/stage1_validation/configs/train_training_config.local.json`

关键参数摘要：

- backbone：
  - `/tmp/qwen2vl2b_local`
- `model_type`：
  - `qwen2_vl`
- LoRA：
  - `r=8`
  - `alpha=16`
- batch size：
  - `1`
- `max_steps`：
  - `1`
- 输出目录：
  - `/tmp/nexus_stage1_config_train`

### 3. 执行方式

执行命令形态为：

```bash
python -m Nexus.training.embedder.multimodal_retrieval \
  --model_config experiments/stage1_validation/configs/train_model_config.local.json \
  --data_config experiments/stage1_validation/configs/train_data_config.local.json \
  --training_config experiments/stage1_validation/configs/train_training_config.local.json
```

### 4. 输出文件

运行摘要：

- `experiments/stage1_validation/config_run_summaries/config_train_smoke_result.json`

训练输出目录：

- `/tmp/nexus_stage1_config_train`

目录内关键文件：

- `adapter_config.json`
- `adapter_model.safetensors`
- `preprocessor_config.json`
- `tokenizer_config.json`
- `video_preprocessor_config.json`

### 5. 关键结果

来自 `config_train_smoke_result.json`：

- `status = success`
- `cuda_visible_devices = 6`
- `train_loss = 0.57421875`
- `train_runtime = 0.9778`

### 6. 这个实验能证明什么

它说明：

- 配置文件模式不是“纸面支持”
- 真正准备第二阶段训练时，已经可以基于 JSON 配置进行管理
- 训练产物结构也已经整理成可复用的标准目录

---

## E6. 配置文件模式评测 smoke

### 1. 实验目的

验证：

- 配置化训练产物能直接被配置化评测入口读取
- train config 和 eval config 之间没有断裂

### 2. 输入配置

模型配置：

- `experiments/stage1_validation/configs/eval_model_adapter.local.json`

评测配置：

- `experiments/stage1_validation/configs/eval_config.local.json`

关键参数摘要：

- `embedder_name_or_path`：
  - `/tmp/nexus_stage1_config_train`
- `dataset_dir`：
  - `examples/multimodal_retrieval/data/eval`
- `cache_path`：
  - `/tmp/nexus_stage1_config_eval_hf_cache`
- `eval_output_dir`：
  - `/tmp/nexus_stage1_config_eval_results`

### 3. 执行方式

执行命令形态为：

```bash
python -m Nexus.evaluation.multimodal_retrieval \
  --eval_config experiments/stage1_validation/configs/eval_config.local.json \
  --model_config experiments/stage1_validation/configs/eval_model_adapter.local.json
```

### 4. 输出文件

摘要：

- `experiments/stage1_validation/config_run_summaries/config_eval_adapter_summary.md`

指标：

- `experiments/stage1_validation/config_run_summaries/config_eval_adapter_eval_results.json`

### 5. 关键结果

来自 `config_eval_adapter_eval_results.json`：

- `ndcg_at_10 = 1.0`
- `recall_at_10 = 1.0`
- `mrr_at_10 = 1.0`

### 6. 这个实验能证明什么

它说明：

- 第一阶段已经把“配置化训练 -> 配置化评测”的链条打通
- 以后做第二阶段正式训练时，不需要靠手工改 Python 代码切换实验

---

## E7. 真实 public train 子集准备

### 1. 实验目的

验证：

- manifest 驱动的数据准备工具不是只对 toy 数据生效
- 对真实 MMEB public train 子集也能转换成 Nexus 训练格式

### 2. 输入数据

真实原始数据根目录：

- `/tmp/public_mmeb_raw`

本轮成功转换的来源：

- `TIGER-Lab/MMEB-train / HatefulMemes / original`

### 3. 执行方式

运行入口：

- `tools/multimodal_retrieval/prepare_mmeb_v2_train_data.py`

执行形态：

```bash
python tools/multimodal_retrieval/prepare_mmeb_v2_train_data.py \
  --raw-root /tmp/public_mmeb_raw \
  --output-root experiments/stage1_validation/data_preparation/train_public_subset \
  --write-stage-configs-dir experiments/stage1_validation/data_preparation/train_public_subset_configs \
  --overwrite
```

### 4. 输出文件

转换摘要：

- `experiments/stage1_validation/data_preparation/train_public_subset/conversion_summary.json`

转换后的训练文件：

- `experiments/stage1_validation/data_preparation/train_public_subset/image/HatefulMemes.jsonl`
- `experiments/stage1_validation/data_preparation/train_public_subset/visdoc/HatefulMemes.jsonl`

stage 配置：

- `experiments/stage1_validation/data_preparation/train_public_subset_configs/stage_a_data_config.json`
- `experiments/stage1_validation/data_preparation/train_public_subset_configs/stage_b_data_config.json`
- `experiments/stage1_validation/data_preparation/train_public_subset_configs/stage_c_data_config.json`

### 5. 关键结果

来自 `conversion_summary.json`：

- `image/HatefulMemes`：
  - `count = 8500`
- `visdoc/HatefulMemes`：
  - `count = 8500`

### 6. 这个实验能证明什么

它说明：

- 第一阶段整理的不只是模型侧代码
- 训练数据准备工具已经能处理真实 public 数据
- 后续和负责数据收集的同学对接时，不会停留在“格式怎么转都还不清楚”的状态

---

## E8. MMEB eval prep dry-run：覆盖 image / video / visdoc

### 1. 实验目的

验证：

- `prepare_mmeb_v2_eval_data.py` 能够覆盖 MMEB 的不同模态
- 能够为不同模态生成正确的转换命令和 `eval_config.json`

### 2. 覆盖的数据集

- image：
  - `HatefulMemes`
- video：
  - `MSVD`
- visdoc：
  - `ViDoRe_arxivqa`

### 3. 执行方式

运行入口：

- `tools/multimodal_retrieval/prepare_mmeb_v2_eval_data.py`

执行形态：

```bash
python tools/multimodal_retrieval/prepare_mmeb_v2_eval_data.py \
  --raw-root experiments/stage1_validation/data_preparation/eval_prep/raw_eval_placeholder \
  --output-root experiments/stage1_validation/data_preparation/eval_prep/prepared_eval \
  --write-eval-configs-dir experiments/stage1_validation/data_preparation/eval_prep/eval_configs \
  --dry-run \
  --overwrite
```

### 4. 输出文件

总报告：

- `experiments/stage1_validation/data_preparation/eval_prep/dry_run_report.json`

生成的评测配置：

- `experiments/stage1_validation/data_preparation/eval_prep/eval_configs/HatefulMemes.eval_config.json`
- `experiments/stage1_validation/data_preparation/eval_prep/eval_configs/MSVD.eval_config.json`
- `experiments/stage1_validation/data_preparation/eval_prep/eval_configs/ViDoRe_arxivqa.eval_config.json`

### 5. 关键结果

`dry_run_report.json` 中可以看到：

- image 任务会从 `MMEB_Test_Instruct` 读取 metadata
- video 任务会把 `frame_root` 作为 eval 时的 `image_root`
- visdoc 任务会为 `ViDoRe_arxivqa` 生成独立转换命令和 eval config

### 6. 这个实验能证明什么

它说明：

- 评测准备工具已经不是只面向某一个固定任务写死
- 至少在 image / video / visdoc 三类代表任务上，配置与路径逻辑已经整理出来

---

## E9. 真实 MMEB 子集转换与 end-to-end 评测

### 1. 实验目的

这是第一阶段最关键的真实性实验之一。目的不是 dry-run，而是确认：

- MMEB 风格真实数据可以被转换成 Nexus local-eval 格式
- 转换后的数据可以被 eval runner 直接评测
- 真实 visdoc 子集会不会暴露出之前 toy 数据根本碰不到的问题

### 2. 选择的数据集

- `ViDoRe_arxivqa`

选择原因：

- 它属于 MMEB 的 visdoc 子集，能覆盖最复杂的一类输入
- metadata 采用的是 BEIR-configs 结构
- 真实转换时会逼出 image / instruction / corpus / qrels 的兼容性问题

### 3. 真实暴露并修复的问题

这条实验链路实际逼出了三类重要问题：

1. MMEB image-family eval metadata 不一定是最初假设的 pair-style schema
2. `ViDoRe_*` 使用的是 BEIR-configs 结构
3. Qwen2-VL 在“带图输入 + `truncation=max_length`”时可能报 image token mismatch

这三个问题都已经在代码中修复，并补了测试。

### 4. 执行方式

#### 4.1 数据转换

运行入口：

- `tools/multimodal_retrieval/prepare_mmeb_v2_eval_data.py`

执行形态：

```bash
PYTHONPATH=/home/szn/zhangx/explore/Nexus \
/home/szn/zht/miniconda3/envs/costa/bin/python \
tools/multimodal_retrieval/prepare_mmeb_v2_eval_data.py \
  --raw-root experiments/stage1_validation/mmeb_real_eval/raw \
  --output-root experiments/stage1_validation/mmeb_real_eval/prepared_eval \
  --datasets ViDoRe_arxivqa \
  --max-rows-per-dataset 2 \
  --python-bin /home/szn/zht/miniconda3/envs/costa/bin/python \
  --write-eval-configs-dir experiments/stage1_validation/mmeb_real_eval/eval_configs \
  --overwrite
```

#### 4.2 真实评测

在确认空闲 GPU 后，使用空闲 `GPU 2` 运行：

```bash
CUDA_VISIBLE_DEVICES=2 \
/home/szn/zht/miniconda3/envs/costa/bin/python \
-m Nexus.evaluation.multimodal_retrieval \
  --eval_config /home/szn/zhangx/explore/Nexus/experiments/stage1_validation/mmeb_real_eval/eval_configs/ViDoRe_arxivqa.eval_config.json \
  --model_config /home/szn/zhangx/explore/Nexus/experiments/stage1_validation/configs/eval_model_base.local.json
```

### 5. 输出文件

评测配置：

- `experiments/stage1_validation/mmeb_real_eval/eval_configs/ViDoRe_arxivqa.eval_config.json`

转换结果：

- `experiments/stage1_validation/mmeb_real_eval/prepared_eval/ViDoRe_arxivqa/corpus.jsonl`
- `experiments/stage1_validation/mmeb_real_eval/prepared_eval/ViDoRe_arxivqa/test_queries.jsonl`
- `experiments/stage1_validation/mmeb_real_eval/prepared_eval/ViDoRe_arxivqa/test_qrels.jsonl`
- `experiments/stage1_validation/mmeb_real_eval/prepared_eval/ViDoRe_arxivqa/dataset_meta.json`
- `experiments/stage1_validation/mmeb_real_eval/prepared_eval/conversion_summary.json`

评测结果：

- `experiments/stage1_validation/mmeb_real_eval/prepared_eval/_results/ViDoRe_arxivqa/summary.md`
- `experiments/stage1_validation/mmeb_real_eval/prepared_eval/_results/ViDoRe_arxivqa/qwen2vl2b_local/NoReranker/EVAL/eval_results.json`
- `experiments/stage1_validation/mmeb_real_eval/prepared_eval/_results/ViDoRe_arxivqa/qwen2vl2b_local/NoReranker/ViDoRe_arxivqa-test.json`

### 6. 关键结果

来自 `summary.md`：

- `ndcg_at_10 = 81.546`
- `recall_at_10 = 100.000`

### 7. 这个实验能证明什么

它说明：

- 我们的 codebase 整理已经不只是“本地 toy demo 能跑”
- 至少有一条真实 MMEB 子集链路已经被打通
- 这条链路还真实帮助我们发现并修复了之前没有暴露出来的系统性问题

---

## E10. backbone matrix 验证：当前 `costa` 环境

### 1. 实验目的

验证：

- 当前环境里，Nexus 声称支持的 backbone family 到底哪些是真的能加载
- 把“代码分支存在”和“真实可加载”明确区分开

### 2. 执行方式

使用脚本：

- `tools/multimodal_retrieval/validate_backbone_matrix.py`

执行命令：

```bash
PYTHONPATH=/home/szn/zhangx/explore/Nexus \
/home/szn/zht/miniconda3/envs/costa/bin/python \
tools/multimodal_retrieval/validate_backbone_matrix.py \
  --output-dir experiments/stage1_validation/backbone_matrix/costa \
  --label costa
```

### 3. 输出文件

- `experiments/stage1_validation/backbone_matrix/costa/report.json`
- `experiments/stage1_validation/backbone_matrix/costa/summary.md`

### 4. 关键结果

当前环境：

- Python：
  - `/home/szn/zht/miniconda3/envs/costa/bin/python`
- `transformers = 4.52.3`

结果：

- `qwen2_vl`：loaded
- `qwen2_5_vl`：loaded
- `llava_next`：loaded
- `qwen3_vl`：unavailable

这里的 `unavailable` 不是代码没写，而是因为：

- 当前 `transformers 4.52.3` 没有 `Qwen3VLForConditionalGeneration`

### 5. 这个实验能证明什么

它说明：

- 我们没有把“环境里缺类”硬说成“已经支持”
- 当前 codebase 对 family 的支持边界已经被客观记录下来

---

## E11. backbone matrix 验证：隔离 `transformers 4.57.3` 环境

### 1. 实验目的

因为 `costa` 环境里的 `transformers` 版本不足以覆盖 `qwen3_vl`，所以需要额外做一次隔离环境验证，确认：

- `qwen3_vl` 真的是代码可加载，而不是文档里写着支持

### 2. 实验环境

隔离环境：

- `/tmp/nexus_stage1_tf457_env`

版本：

- `transformers = 4.57.3`

### 3. 执行方式

执行命令：

```bash
PYTHONPATH=/home/szn/zhangx/explore/Nexus \
/tmp/nexus_stage1_tf457_env/bin/python \
tools/multimodal_retrieval/validate_backbone_matrix.py \
  --output-dir experiments/stage1_validation/backbone_matrix/transformers_4_57_3 \
  --label transformers_4_57_3 \
  --fail-on-unavailable \
  --fail-on-failure
```

### 4. 输出文件

- `experiments/stage1_validation/backbone_matrix/transformers_4_57_3/report.json`
- `experiments/stage1_validation/backbone_matrix/transformers_4_57_3/summary.md`

### 5. 关键结果

四个 family 全部通过：

- `qwen2_vl`
- `qwen2_5_vl`
- `qwen3_vl`
- `llava_next`

### 6. 这个实验能证明什么

它说明：

- 如果第二阶段最终选 `Qwen3-VL`，当前代码底座并不会成为阻碍
- 真正需要做的是把正式训练环境升级到合适版本

---

## E12. `qwen3_vl` base toy eval

### 1. 实验目的

在已经验证 family-loader 可用的基础上，进一步确认：

- 真正的 `Qwen3-VL-2B-Instruct` 权重能够被 Nexus 评测入口加载
- 不只是能 `from_pretrained`，而且能在 toy retrieval 数据上真实跑完一次评测

### 2. 实验环境

- 模型：
  - `/tmp/qwen3vl2b_instruct_local`
- Python：
  - `/tmp/nexus_stage1_tf457_env/bin/python`
- `transformers`：
  - `4.57.3`
- 物理 GPU：
  - `2`

### 3. 执行方式

执行命令：

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/szn/zhangx/explore/Nexus \
/tmp/nexus_stage1_tf457_env/bin/python \
-m Nexus.evaluation.multimodal_retrieval \
  --eval_config /home/szn/zhangx/explore/Nexus/experiments/stage1_validation/qwen3_vl_full_loop/configs/eval_config_toy.qwen3.local.json \
  --model_config /home/szn/zhangx/explore/Nexus/experiments/stage1_validation/qwen3_vl_full_loop/configs/eval_model_base.qwen3.local.json
```

### 4. 输出文件

- `experiments/stage1_validation/qwen3_vl_full_loop/results/toy_eval/summary.md`
- `experiments/stage1_validation/qwen3_vl_full_loop/results/toy_eval/qwen3vl2b_instruct_local/NoReranker/EVAL/eval_results.json`

### 5. 关键结果

- `ndcg_at_10 = 1.0`
- `recall_at_10 = 1.0`

### 6. 这个实验能证明什么

它说明：

- `qwen3_vl` 在 Nexus 里的评测入口已经真实可用
- 问题已经不再是“能否识别类”，而是真正跑通了模型评测

---

## E13. `qwen3_vl` 训练 smoke

### 1. 实验目的

验证：

- `qwen3_vl` 在 Nexus 当前训练入口中能否真正完成一次 LoRA smoke 训练
- 不只是能加载 base model，而是训练链路也能走通

### 2. 输入配置

- `experiments/stage1_validation/qwen3_vl_full_loop/configs/train_model_config.qwen3.local.json`
- `experiments/stage1_validation/qwen3_vl_full_loop/configs/train_data_config.qwen3.local.json`
- `experiments/stage1_validation/qwen3_vl_full_loop/configs/train_training_config.qwen3.local.json`

关键配置：

- `model_name_or_path`：
  - `/tmp/qwen3vl2b_instruct_local`
- `model_type`：
  - `qwen3_vl`
- LoRA：
  - `r=8`
  - `alpha=16`
- `max_steps = 1`
- 输出目录：
  - `/tmp/nexus_stage1_qwen3_config_train`

### 3. 执行方式

执行命令：

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/szn/zhangx/explore/Nexus \
/tmp/nexus_stage1_tf457_env/bin/python \
-m Nexus.training.embedder.multimodal_retrieval \
  --model_config /home/szn/zhangx/explore/Nexus/experiments/stage1_validation/qwen3_vl_full_loop/configs/train_model_config.qwen3.local.json \
  --data_config /home/szn/zhangx/explore/Nexus/experiments/stage1_validation/qwen3_vl_full_loop/configs/train_data_config.qwen3.local.json \
  --training_config /home/szn/zhangx/explore/Nexus/experiments/stage1_validation/qwen3_vl_full_loop/configs/train_training_config.qwen3.local.json
```

### 4. 输出文件

训练输出目录：

- `/tmp/nexus_stage1_qwen3_config_train`

其中关键文件：

- `adapter_config.json`
- `adapter_model.safetensors`
- `preprocessor_config.json`
- `tokenizer_config.json`
- `video_preprocessor_config.json`

汇总文件：

- `experiments/stage1_validation/qwen3_vl_full_loop/run_summary.json`

### 5. 关键结果

实际训练输出：

- `train_loss = 0.57421875`
- `train_runtime = 1.1475`
- `train_steps_per_second = 0.871`

### 6. 这个实验能证明什么

它说明：

- `qwen3_vl` 不只是“能加载能评测”
- 在当前 Nexus 训练代码里也已经能走通完整的 LoRA smoke 训练

---

## E14. `qwen3_vl` runtime validation：base + adapter 推理与本地评测闭环

### 1. 实验目的

验证：

- `qwen3_vl` 的基础模型推理可用
- `qwen3_vl` 的训练输出目录可被重新加载
- base model 和 adapter model 都能做本地评测

### 2. 执行方式

执行命令：

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/szn/zhangx/explore/Nexus \
/tmp/nexus_stage1_tf457_env/bin/python \
tools/multimodal_retrieval/runtime_validation.py \
  --base-model-path /tmp/qwen3vl2b_instruct_local \
  --adapter-model-path /tmp/nexus_stage1_qwen3_config_train \
  --model-type qwen3_vl \
  --single-device cuda:0 \
  --output-root /home/szn/zhangx/explore/Nexus/experiments/stage1_validation/qwen3_vl_full_loop/runtime_validation
```

### 3. 输出文件

总报告：

- `experiments/stage1_validation/qwen3_vl_full_loop/runtime_validation/runtime_validation_report.json`

摘要：

- `experiments/stage1_validation/qwen3_vl_full_loop/runtime_validation/base_eval/summary.md`
- `experiments/stage1_validation/qwen3_vl_full_loop/runtime_validation/adapter_eval/summary.md`

### 4. 关键结果

来自 `runtime_validation_report.json`：

- base model query embedding shape：
  - `[4, 2048]`
- base model passage embedding shape：
  - `[4, 2048]`
- adapter model query embedding shape：
  - `[4, 2048]`
- adapter model passage embedding shape：
  - `[4, 2048]`

base toy eval：

- `ndcg_at_10 = 1.0`
- `recall_at_10 = 1.0`

adapter toy eval：

- `ndcg_at_10 = 1.0`
- `recall_at_10 = 1.0`

### 5. 这个实验能证明什么

它说明：

- `qwen3_vl` 的完整 smoke 闭环现在已经成立：
  - 训练
  - 保存
  - 重载
  - 推理
  - 本地评测

---

## E15. `qwen3_vl` 真实 MMEB 子集评测

### 1. 实验目的

验证：

- `qwen3_vl` 不是只在 toy 数据上工作
- 它也能直接进入真实 MMEB 子集的评测链路

### 2. 输入配置

- `experiments/stage1_validation/qwen3_vl_full_loop/configs/eval_config_mmeb.qwen3.local.json`
- `experiments/stage1_validation/qwen3_vl_full_loop/configs/eval_model_base.qwen3.local.json`

评测数据仍然使用已经准备好的：

- `ViDoRe_arxivqa`

### 3. 执行方式

执行命令：

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/szn/zhangx/explore/Nexus \
/tmp/nexus_stage1_tf457_env/bin/python \
-m Nexus.evaluation.multimodal_retrieval \
  --eval_config /home/szn/zhangx/explore/Nexus/experiments/stage1_validation/qwen3_vl_full_loop/configs/eval_config_mmeb.qwen3.local.json \
  --model_config /home/szn/zhangx/explore/Nexus/experiments/stage1_validation/qwen3_vl_full_loop/configs/eval_model_base.qwen3.local.json
```

### 4. 输出文件

- `experiments/stage1_validation/qwen3_vl_full_loop/results/mmeb_eval/summary.md`
- `experiments/stage1_validation/qwen3_vl_full_loop/results/mmeb_eval/qwen3vl2b_instruct_local/NoReranker/EVAL/eval_results.json`
- `experiments/stage1_validation/qwen3_vl_full_loop/results/mmeb_eval/qwen3vl2b_instruct_local/NoReranker/ViDoRe_arxivqa-test.json`

### 5. 关键结果

- `ndcg_at_10 = 81.546`
- `recall_at_10 = 100.000`

### 6. 这个实验能证明什么

它说明：

- `qwen3_vl` 现在不仅支持本地 smoke
- 也已经接上了真实 MMEB 子集评测链路

---

## E16. example 与 config 鲁棒性修复验证

### 1. 为什么要做这个实验

在 `qwen3_vl` 真正跑通之后，我又额外检查了 example 层，发现了三类会影响后续同学复现的隐患：

1. 示例 `model_config.json` 写死 `qwen2_vl`
2. inference example 使用了仓库里根本不存在的图片路径
3. config-file 模式里的相对路径依赖当前工作目录，而不是相对配置文件

这些问题不修的话，会出现一种很糟糕的情况：

- 核心代码实际上已经支持了
- 但同学按 example 跑，反而先死在错误示例上

### 2. 已做修复

代码修复点包括：

- `examples/multimodal_retrieval/training/model_config.json`
- `examples/multimodal_retrieval/evaluation/model_config.json`
- `examples/multimodal_retrieval/inference/inference_multimodal_embedder.py`
- `examples/multimodal_retrieval/inference/encode_demo.py`
- `Nexus/abc/arguments.py`
- `Nexus/evaluation/multimodal_retrieval/arguments.py`
- `tests/multimodal_retrieval/test_cli_config.py`

具体修复内容：

- example config 的 `model_type` 改成 `auto`
- inference example 改用仓库真实存在的 `query.ppm / doc.ppm / neg.ppm`
- inference example 增加 `MODEL_NAME_OR_PATH / PROCESSOR_NAME_OR_PATH / MODEL_TYPE` 环境变量覆盖
- config-file JSON 中的相对路径改为相对配置文件目录解析

### 3. 回归验证

再次执行：

```bash
PYTHONPATH=/home/szn/zhangx/explore/Nexus \
/home/szn/zht/miniconda3/envs/costa/bin/python -m pytest tests/multimodal_retrieval -q
```

结果：

- `37 passed`

再次执行：

```bash
PYTHON_BIN=/home/szn/zht/miniconda3/envs/costa/bin/python \
bash tools/multimodal_retrieval/validate_stack.sh
```

结果：

- 全链路通过

### 4. 额外示例验证

为了证明 example 真的修好了，而不是只改了文件内容，我从仓库外部目录 `/tmp` 直接运行：

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/szn/zhangx/explore/Nexus \
MODEL_NAME_OR_PATH=/tmp/qwen3vl2b_instruct_local \
PROCESSOR_NAME_OR_PATH=/tmp/qwen3vl2b_instruct_local \
MODEL_TYPE=qwen3_vl \
/tmp/nexus_stage1_tf457_env/bin/python \
/home/szn/zhangx/explore/Nexus/examples/multimodal_retrieval/inference/inference_multimodal_embedder.py
```

输出分数：

```text
tensor([[0.9532, 0.9368],
        [0.8954, 0.9024]])
```

以及：

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/szn/zhangx/explore/Nexus \
MODEL_NAME_OR_PATH=/tmp/qwen3vl2b_instruct_local \
PROCESSOR_NAME_OR_PATH=/tmp/qwen3vl2b_instruct_local \
MODEL_TYPE=qwen3_vl \
/tmp/nexus_stage1_tf457_env/bin/python \
/home/szn/zhangx/explore/Nexus/examples/multimodal_retrieval/inference/encode_demo.py
```

输出：

```text
query shape: (1, 2048)
doc shape: (2, 2048)
scores: [[0.9630636 0.952819 ]]
```

### 5. 这个实验能证明什么

它说明：

- 现在 example 脚本也已经真实可用
- 而且不会再因为 cwd 或错误示例路径先失败
- 这对第一阶段“可交接、可复现”的价值非常大

---

## 五、这些实验合起来能说明什么

如果把所有实验放在一起看，可以得到一个比较硬的结论：

### 1. 微调能力不是纸面支持

证据：

- E5 配置化训练 smoke 成功
- E4、E6 又证明训练产物可以继续被推理和评测使用

### 2. 推理能力不是只支持单卡 demo

证据：

- E3 验证了多设备一致性，数值误差为 `0.0`

### 3. 评测能力不是只支持 toy 样例

证据：

- E9 已经完成真实 MMEB 子集 `ViDoRe_arxivqa` 的 end-to-end eval

### 4. 数据工具链已经具备真实可用性

证据：

- E7 真实 public train 子集成功转换
- E8 覆盖了 image / video / visdoc 三类 eval prep
- E9 进一步把 visdoc 的真实转换和评测跑通

### 5. backbone 兼容不是“先写上再说”

证据：

- E10、E11 对四个 family 做了真实 checkpoint 加载验证
- 同时把环境版本边界明确暴露出来，没有混淆“代码支持”和“当前环境可用”

### 6. `qwen3_vl` 现在已经不是“只做了 family 兼容”

证据：

- E12 跑通了 `qwen3_vl` base toy eval
- E13 跑通了 `qwen3_vl` LoRA smoke 训练
- E14 跑通了 `qwen3_vl` adapter 重载、推理和本地评测
- E15 跑通了 `qwen3_vl` 在真实 MMEB 子集上的评测
- E16 又把 qwen3 会真正用到的 example/config 层隐患补掉并验证

---

## 六、当前仍然保留的诚实边界

虽然第一阶段已经可以认为完成，但下面这些话现在还不能说：

- 不能说“第二阶段正式训练已经完成”
- 不能说“所有 backbone 都已经做完完整 benchmark 闭环”
- 不能说“已经超过 Qwen3-VL-Embedding-8B”

目前最准确的说法是：

- 第一阶段 codebase 整理已经完成
- 并且已经经过了多层真实实验验证
- 当前已经完成 smoke 级完整闭环验证的 backbone 包括：
  - `Qwen2-VL-2B-Instruct`
  - `Qwen3-VL-2B-Instruct`
- `Qwen2.5-VL / Llava-Next` 已完成真实 family-loader 验证

---

## 七、接下来的补充实验计划

本轮 `qwen3_vl` 全闭环实验已经完成。后续如果继续推进，更值得做的方向是：

- 把 `Qwen2.5-VL` 也补成同等级 smoke 闭环
- 在正式训练环境中用更大规模 qwen3 backbone 做第二阶段 recipe 验证
