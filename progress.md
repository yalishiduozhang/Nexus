# 项目进展记录

更新时间：2026-03-25

本文档只记录已经真实发生并确认过的工作，不把“计划做”“可能做”写成“已经完成”。

## 2026-03-24

### 一、前置审查与方向确认

已完成：

- 审查 `Nexus`、`FlagEmbedding`、`VLM2Vec`，确定整体集成方向。
- 明确本项目采用的工程思路：
  - 以 `Nexus` 为主仓库
  - 参考 `FlagEmbedding` 的 train / inference / evaluation 分层方式
  - 面向 MMEB v2 的真实任务形态补齐工具和链路

### 二、多模态 embedding 主链路接入

已完成：

- 在 `Nexus` 中新增多模态 embedding 训练链路。
- 在 `Nexus` 中新增多模态 embedding 推理链路。
- 在 `Nexus` 中新增多模态 embedding 评测链路。
- 新增 `examples/multimodal_retrieval/` 示例配置、示例脚本和示例数据。

对应里程碑提交：

- `8a5d364` `Add multimodal embedding pipeline and examples`

### 三、面向 MMEB v2 的能力补齐

已完成：

- 增强多模态字段归一化，支持 image / video / pages 等常见输入形式。
- 增加原始视频支持和视频相关 chat-template 兼容。
- 增加 `Qwen3-VL` 相关兼容注册。
- 让评测支持 `media_root / image_root / video_root`。
- 修复多设备推理只使用首卡的问题。

### 四、数据工具与文档骨架

已完成：

- 新增项目级文档：
  - `plan.md`
  - `progress.md`
- 新增数据相关资产：
  - `docs/multimodal_retrieval/MMEB_v2_data_inventory.md`
  - `docs/multimodal_retrieval/MMEB_v2_manifest.json`
  - `tools/multimodal_retrieval/convert_vlm2vec_train_to_nexus.py`
  - `tools/multimodal_retrieval/convert_vlm2vec_eval_to_nexus.py`
  - `tools/multimodal_retrieval/export_mmeb_v2_inventory.py`
  - `tools/multimodal_retrieval/export_mmeb_v2_manifest.py`
  - `tools/multimodal_retrieval/README.md`

### 五、环境与共享 GPU 安全工具

已完成：

- 新增：
  - `tools/multimodal_retrieval/check_idle_gpus.py`
  - `tools/multimodal_retrieval/create_conda_env.sh`
  - `tools/multimodal_retrieval/environment.yml`
  - `docs/multimodal_retrieval/training_recipe.md`
  - `docs/multimodal_retrieval/data_collection_playbook.md`
- 更新示例训练脚本，要求共享 GPU 运行时显式设置 `CUDA_VISIBLE_DEVICES`。

### 六、可选依赖解耦与测试

已完成：

- 通过 lazy import 降低与当前任务无关依赖的强耦合：
  - `onnx`
  - `onnxruntime`
  - `tensorrt`
- 增加多模态回归测试，覆盖：
  - 视频字段归一化
  - 前缀 query / pos 解析
  - 评测媒体根路径解析
  - 转换脚本关键辅助逻辑
- 新增 `tools/multimodal_retrieval/validate_stack.sh`，把关键验证流程固化成一键脚本。

### 七、当日验证结果

已完成：

- `python -m py_compile` 覆盖所有触及的 Python 文件。
- `pytest tests/multimodal_retrieval -q` 通过。
- `validate_stack.sh` 在隔离环境中通过。
- `import Nexus` 及多模态模块导入成功。

## 2026-03-25

### 一、manifest 驱动的数据准备工具继续补齐

已完成：

- 新增：
  - `tools/multimodal_retrieval/vlm2vec_manifest_lib.py`
  - `tools/multimodal_retrieval/hf_dataset_manager.py`
  - `tools/multimodal_retrieval/prepare_public_data.py`
  - `tools/multimodal_retrieval/prepare_mmeb_v2_train_data.py`
- 在多模态 train / eval 入口中增加 split-json 配置文件解析能力。
- 补充本地 bundled 示例 train / eval 数据。
- 在转换工具中增加 Git LFS pointer 检测，避免元数据克隆导致低层 parquet 报错。

### 二、数据工具细节修复

已完成：

- 修复本地输入目录递归扫描，只扫顶层的问题。
- 修复旧 manifest 缺少下载元信息时的兼容处理。
- 修复 Hugging Face 数据集 URL 规划逻辑。
- 为 `check_idle_gpus.py` 增加 `--input` 快照输入模式。

### 三、真实 public data smoke

已完成：

- 使用真实 MMEB train 子集做本地 smoke：
  - 来源：`TIGER-Lab/MMEB-train / HatefulMemes / original`
- 成功下载并转换为 Nexus 训练格式。
- 成功生成 stage 训练配置。

### 四、MMEB v2 eval-prep 工具

已完成：

- 新增 `tools/multimodal_retrieval/prepare_mmeb_v2_eval_data.py`
- 为 image / video / visdoc 代表数据集完成 dry-run 验证。
- 更新 `tools/multimodal_retrieval/README.md` 与 `validate_stack.sh`。

对应验证：

- `HatefulMemes`
- `MSVD`
- `ViDoRe_arxivqa`

### 五、本地离线 backbone smoke 训练

已完成：

- 本地离线下载并校验 `Qwen/Qwen2-VL-2B-Instruct` 到 `/tmp/qwen2vl2b_local`
- 修复真实 smoke 训练过程中暴露出的兼容性问题：
  - `Accelerator.unwrap_model(..., keep_torch_compile=...)` 兼容补丁
  - 多模态 `media_root` / `image_root` / `video_root` 解析修复
  - Qwen2-VL visual batching 在 processor 缺少 `.pad()` 时的兼容路径
  - 示例图片尺寸过小导致预处理失败的问题
  - 示例训练数据的媒体路径修复

结果：

- 本地离线 `Qwen2-VL-2B-Instruct` one-step LoRA smoke finetune 成功
- 输出目录成功写出
- 报告 `train_loss = 0.474609375`

对应提交：

- `6f534aa` `Validate offline Qwen2-VL smoke training`

### 六、第一阶段全面验收与运行时修复

已完成：

- 建立专门验收目录：
  - `experiments/stage1_validation/`
- 在真实运行时暴露并修复的问题包括：
  - LoRA 输出目录不能直接被加载
  - adapter 目录缺少 processor 时不能自动回退到 base model processor
  - 已是 PEFT 模型时重复套 LoRA
  - `bf16` 推理输出直接转 numpy 失败
  - `faiss` 顶层强依赖导致评测路径不可用
  - `pytrec_eval` 顶层强导入导致部分工具 / 测试导入失败
  - 多进程 embedder 结束时的 semaphore warning
  - `nvidia-smi --format=csv,noheader,nounits` 在当前环境中的兼容性问题

相关新增或重点变更：

- `Nexus/modules/multimodal.py`
- `Nexus/training/embedder/multimodal_retrieval/modeling.py`
- `Nexus/inference/embedder/multimodal_retrieval/generic.py`
- `Nexus/evaluation/text_retrieval/utils.py`
- `Nexus/abc/inference/embedder/AbsEmbedder.py`
- `tools/multimodal_retrieval/runtime_validation.py`
- `tests/multimodal_retrieval/test_multimodal_utils.py`
- `tests/multimodal_retrieval/test_data_tools.py`

### 七、第一阶段真实验收结果

已确认：

- `pytest tests/multimodal_retrieval -q` 通过，结果为 `31 passed`
- `tools/multimodal_retrieval/validate_stack.sh` 通过
- 使用空闲 GPU `4,5` 完成基础模型与多设备推理一致性验证
- 使用空闲 GPU `6` 完成配置文件模式训练 smoke
- 基础模型和 LoRA 输出目录都已完成本地评测
- 配置化训练产物已完成重新加载与评测
- 真实 public train 子集准备成功写入 `experiments/stage1_validation/data_preparation/`
- MMEB eval prep dry-run 成功写入 `experiments/stage1_validation/data_preparation/eval_prep/`

关键结果：

- embedding 维度：`1536`
- 多设备一致性：
  - `query_allclose = true`
  - `passage_allclose = true`
  - `max_query_abs_diff = 0.0`
  - `max_passage_abs_diff = 0.0`
- toy eval：
  - `ndcg_at_10 = 1.0`
  - `recall_at_10 = 1.0`

### 八、环境与汇报文档补强

已完成：

- 新增：
  - `experiments/stage1_validation/environment_and_backbone.md`
- 该文档专门说明：
  - 当前代码层支持哪些 backbone family
  - 当前真实完成闭环验证的是哪个 backbone
  - 老师尚未唯一指定最终 backbone 对第一阶段和第二阶段分别有什么影响
- 更新 `tools/multimodal_retrieval/create_conda_env.sh`，使其在新建隔离环境后默认执行：
  - 依赖安装
  - 版本打印
  - `validate_stack.sh`
- 更新 `tools/multimodal_retrieval/README.md`，把标准路径明确成“隔离环境安装 + 验证”
- 更新中文汇报总文档：
  - `/home/szn/zhangx/explore/docs/Nexus_多模态Embedding当前进展详解.md`
  - `/home/szn/zhangx/explore/docs/Nexus_多模态Embedding当前进展详解_2026-03-25.md`

### 九、当前明确的边界与未决事项

当前可以真实确认的边界：

- `Qwen2-VL-2B-Instruct` 已完成真实 smoke 闭环验证
- `Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Llava-Next` 已完成代码层适配
- 但除 `Qwen2-VL-2B-Instruct` 外，其他 backbone 目前尚未逐个做同等级别真实验收

当前仍建议后续继续确认的事项：

- 老师第二阶段最终偏好的 backbone family 与参数规模
- 更大规模数据混合与正式训练资源预算
- `HatefulMemes` 在 manifest 中同时出现在 `image` 与 `visdoc` 的模态归类问题

### 十、第一阶段收尾补强与真实 MMEB 子集闭环

已完成：

- 修复 `create_conda_env.sh` 的仓库路径鲁棒性
- 修复 `validate_stack.sh` 对机器私有 `VLM2Vec` 路径的依赖
- 修复真实 MMEB visdoc 子集转换所需的 instruction-style / BEIR-configs 兼容
- 修复 Qwen2-VL 带图输入在 `truncation=max_length` 下的 image token mismatch
- 修复 `prepare_mmeb_v2_eval_data.py` 默认把缓存写到仓库根目录的问题
- 为 `prepare_mmeb_v2_eval_data.py` 增加 `--write-configs-only`
- 修复 eval 数据覆盖写入时“失败先删旧数据”的风险，改成 staging 安全替换
- 新增 `tools/multimodal_retrieval/validate_backbone_matrix.py`

新增真实验证：

- `pytest tests/multimodal_retrieval -q` 重新通过，结果提升为 `37 passed`
- `tools/multimodal_retrieval/validate_stack.sh` 重新通过
- 使用空闲 `GPU 2` 完成 `ViDoRe_arxivqa` 真实 MMEB 子集 end-to-end eval
- 在 `costa` 环境下完成：
  - `qwen2_vl`
  - `qwen2_5_vl`
  - `llava_next`
  的 tiny-checkpoint 加载验证
- 在额外隔离的 `transformers 4.57.3` 环境下完成：
  - `qwen3_vl`
  的真实加载验证

关键结果：

- 真实 MMEB 子集 `ViDoRe_arxivqa`：
  - `ndcg_at_10 = 81.546`
  - `recall_at_10 = 100.000`

对应新增产物：

- `experiments/stage1_validation/mmeb_real_eval/`
- `experiments/stage1_validation/backbone_matrix/`

更新后的最准确结论：

- 第一阶段代码底座已经完成
- `Qwen2-VL-2B-Instruct` 已完成完整闭环验证
- 四个 backbone family 都完成了真实加载验证
- 第二阶段正式训练仍需等待最终 backbone 决策和训练资源方案
