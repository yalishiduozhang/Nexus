# 第一阶段全面验收结果

更新时间：2026-03-25

本文档只记录已经真实执行并确认过的结果，不把“计划做”写成“已经完成”。

## 一、当前结论

截至本次验收结束，可以真实确认：

- 第一阶段要求的 `codebase 整理` 已经完成到可交接、可复现、可继续支撑第二阶段的程度。
- Nexus 现在已经具备多模态 embedding 的微调、推理、评测、数据准备、隔离环境和共享 GPU 安全执行能力。
- MMEB 相关链路已经不只是 dry-run：
  - image / video 模态完成了真实配置生成与 dry-run 验证
  - visdoc 模态完成了真实 MMEB 子集 `ViDoRe_arxivqa` 的 end-to-end 评测
- 所声明支持的 backbone family 已经做了真实的加载验证：
  - 在当前 `costa` 环境下，`qwen2_vl / qwen2_5_vl / llava_next` 均完成 tiny-checkpoint `from_pretrained` 加载验证
  - 在额外隔离的 `transformers 4.57.3` 环境下，`qwen3_vl` 也完成了同样的真实加载验证
- `qwen3_vl` 已进一步完成 smoke 级训练、重载、推理、本地评测与真实 MMEB 子集评测

需要严格说明的边界：

- 目前完成 smoke 级“训练 + 推理 + 本地评测 + 真实 MMEB 子集评测”完整闭环的 backbone 已经包括：
  - `Qwen2-VL-2B-Instruct`
  - `Qwen3-VL-2B-Instruct`
- `Qwen2.5-VL / Llava-Next` 目前仍处于真实 family-loader 验证阶段
- 因此，第一阶段可以说“代码底座已经完成并经过真实验收”，但不能夸大成“第二阶段冲榜训练已经完成”

## 二、本轮重点修复的问题

### 1. 环境脚本路径不鲁棒

修复文件：

- `tools/multimodal_retrieval/create_conda_env.sh`

修复内容：

- 改为基于脚本自身位置计算 `REPO_ROOT`
- `pip install -e` 与 `validate_stack.sh` 都使用绝对仓库路径

结果：

- 脚本从任意工作目录执行都能正确安装和验证

### 2. `validate_stack.sh` 写死私有 `VLM2Vec` 路径

修复文件：

- `tools/multimodal_retrieval/validate_stack.sh`
- `tools/multimodal_retrieval/vlm2vec_manifest_lib.py`
- `tools/multimodal_retrieval/export_mmeb_v2_inventory.py`
- `tools/multimodal_retrieval/export_mmeb_v2_manifest.py`

修复内容：

- 支持优先读取 `VLM2VEC_ROOT`
- 支持自动发现相邻目录 `../VLM2Vec` 和 `../vlm2vec`
- 缺少本地 `VLM2Vec` 时默认跳过 inventory export，不再直接崩溃

结果：

- 工具从“依赖某台机器上的私有路径”变成了“可在其他机器上直接复用”

### 3. 之前没有真实 MMEB 子集 end-to-end 评测

修复文件：

- `tools/multimodal_retrieval/convert_vlm2vec_eval_to_nexus.py`
- `tools/multimodal_retrieval/prepare_mmeb_v2_eval_data.py`
- `Nexus/modules/multimodal.py`

本次真实暴露并修复的问题包括：

- MMEB image-family eval metadata 不是最初假设的 pair-style schema
- `ViDoRe_*` 这类 visdoc 子集实际使用的是 BEIR-configs 结构
- Qwen2-VL 在“带图输入 + `truncation=max_length`”时会触发 image token mismatch

结果：

- `ViDoRe_arxivqa` 现在已经可以被真实转换并完成真实评测

### 4. 评测配置和缓存目录污染仓库根目录

修复文件：

- `tools/multimodal_retrieval/prepare_mmeb_v2_eval_data.py`
- `tools/multimodal_retrieval/prepare_mmeb_v2_train_data.py`
- `.gitignore`

修复内容：

- 新生成的 train/eval 配置会把 `cache_path` 收敛到实验目录内部
- 根目录 `.cache/` 被加入 `.gitignore`

结果：

- 过程性缓存不再默认散落到仓库根目录

### 5. eval 重新生成配置时会被网络环境绑死

修复文件：

- `tools/multimodal_retrieval/prepare_mmeb_v2_eval_data.py`

修复内容：

- 新增 `--write-configs-only`
- 支持只重写 `eval_config.json`，不重新拉取元数据、不重新做转换

结果：

- 后续维护评测配置时不再必须依赖网络

### 6. eval 覆盖写入策略不安全

修复文件：

- `tools/multimodal_retrieval/prepare_mmeb_v2_eval_data.py`

真实暴露的问题：

- 如果在 `--overwrite` 模式下重新转换，而远端元数据拉取失败，旧的 `prepared_eval/<dataset>` 会先被删掉，再整体失败

修复内容：

- 改成先写到临时目录 `.tmp_convert`
- 只有转换成功后才原子替换旧目录

结果：

- 重新转换失败时，旧的已准备数据不会再被意外删掉

### 7. backbone family 验证缺少成体系的产物

新增文件：

- `tools/multimodal_retrieval/validate_backbone_matrix.py`

新增能力：

- 自动检测当前环境里各 family 对应的 `transformers` 类是否存在
- 为每个 family 生成 tiny local checkpoint
- 走真实 `save_pretrained -> load_multimodal_backbone -> from_pretrained` 路径
- 输出 JSON 和 Markdown 报告

## 三、真实验收结果

### A. 回归测试

执行：

- `pytest tests/multimodal_retrieval -q`

结果：

- `37 passed`

### B. 一键验证脚本

执行：

- `bash tools/multimodal_retrieval/validate_stack.sh`

结果：

- 全链路通过
- 包括：
  - `py_compile`
  - `pytest`
  - inventory / manifest 导出
  - train conversion smoke
  - eval conversion smoke

### C. 运行时训练 / 推理 / 本地评测闭环

使用 backbone：

- `/tmp/qwen2vl2b_local`

已真实完成：

- 基础模型单卡推理
- 基础模型多设备一致性验证
- 基础模型本地 toy eval
- one-step LoRA smoke 训练
- LoRA 输出目录重新加载
- LoRA 输出目录本地 toy eval
- 配置文件模式训练
- 配置文件模式评测

已确认结果：

- embedding 维度：`1536`
- 多设备一致性：
  - `query_allclose = true`
  - `passage_allclose = true`
  - `max_query_abs_diff = 0.0`
  - `max_passage_abs_diff = 0.0`
- toy eval：
  - `ndcg_at_10 = 1.0`
  - `recall_at_10 = 1.0`

### D. 真实 MMEB 子集 end-to-end 评测

真实评测数据集：

- `ViDoRe_arxivqa`

真实评测命令：

- `python -m Nexus.evaluation.multimodal_retrieval --eval_config ...ViDoRe_arxivqa.eval_config.json --model_config ...eval_model_base.local.json`

运行约束：

- 使用空闲 `GPU 2`
- 不占用当前正在跑别人任务的 `GPU 1` 和 `GPU 3`

最终结果：

- `ndcg_at_10 = 81.546`
- `recall_at_10 = 100.000`

结果文件：

- `experiments/stage1_validation/mmeb_real_eval/prepared_eval/_results/ViDoRe_arxivqa/summary.md`

### E. Backbone family 验证矩阵

#### 当前 `costa` 环境

环境：

- Python：`/home/szn/zht/miniconda3/envs/costa/bin/python`
- `transformers = 4.52.3`

结果：

- `qwen2_vl`：loaded
- `qwen2_5_vl`：loaded
- `llava_next`：loaded
- `qwen3_vl`：unavailable

原因：

- 当前 `costa` 环境中的 `transformers 4.52.3` 不包含 `Qwen3VLForConditionalGeneration`

结果文件：

- `experiments/stage1_validation/backbone_matrix/costa/report.json`
- `experiments/stage1_validation/backbone_matrix/costa/summary.md`

#### 隔离 `transformers 4.57.3` 验证环境

环境：

- Python：`/tmp/nexus_stage1_tf457_env/bin/python`
- `transformers = 4.57.3`

结果：

- `qwen2_vl`：loaded
- `qwen2_5_vl`：loaded
- `qwen3_vl`：loaded
- `llava_next`：loaded

结果文件：

- `experiments/stage1_validation/backbone_matrix/transformers_4_57_3/report.json`
- `experiments/stage1_validation/backbone_matrix/transformers_4_57_3/summary.md`

这说明两件事：

- 代码层面对四个 family 的加载逻辑已经真实打通
- 如果第二阶段最终选 `Qwen3-VL`，正式环境必须满足 `transformers>=4.57.3`

## 四、阶段一是否可以认为已经完成

可以。

更准确地说，现在可以向老师汇报：

- 第一阶段里“由我负责的 codebase 整理”已经真实完成
- 并且已经经过单元测试、工具链验证、真实训练 smoke、真实推理评测、真实 MMEB 子集评测、backbone family 加载矩阵等多层验收

如果要更严谨一点，可以用下面这句话：

> 第一阶段的多模态 embedding 代码底座已经在 Nexus 中整理完成，并经过了真实可复现的训练、推理、评测和 MMEB 子集验证；后续第二阶段主要工作将转向正式 backbone 选择、数据混合和大规模训练冲榜。

## 五、当前仍需诚实保留的边界

- 我们还没有完成第二阶段的大规模正式训练
- 还没有提交完整 MMEB v2 leaderboard 成绩
- `Qwen2.5-VL / Llava-Next` 目前完成的是 family-loader 真实验证，而不是完整 benchmark 闭环
- 第二阶段最终 backbone 仍建议和老师进一步确认

## 六、关键产物位置

- 第一阶段计划：
  - `experiments/stage1_validation/plan.md`
- 第一阶段结果：
  - `experiments/stage1_validation/results.md`
- backbone 与环境说明：
  - `experiments/stage1_validation/environment_and_backbone.md`
- 真实 MMEB 子集评测配置与结果：
  - `experiments/stage1_validation/mmeb_real_eval/`
- backbone 矩阵验证：
  - `experiments/stage1_validation/backbone_matrix/`
- 运行时训练 / 推理验证：
  - `experiments/stage1_validation/runtime_outputs/`
- `qwen3_vl` 全闭环实验：
  - `experiments/stage1_validation/qwen3_vl_full_loop/`
