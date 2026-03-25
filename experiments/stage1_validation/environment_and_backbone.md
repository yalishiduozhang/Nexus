# 第一阶段环境与 Backbone 说明

更新时间：2026-03-25

本文档主要回答四个问题：

- 目前这套代码到底真实验证过哪些 backbone
- 现在的环境版本是否会影响 backbone 选择
- 老师没有明确指定最终 backbone，这会不会影响第一阶段
- 如果准备进入第二阶段，环境应该怎么建

## 一、当前真实验证的 backbone 分层

为了避免把“代码里写了兼容分支”和“真的验证过”混为一谈，这里把验证强度分成两层。

### 第一层：完整闭环验证

当前真正完成了“训练 / 推理 / 评测 / 真实 MMEB 子集评测”完整闭环验证的 backbone 是：

- `Qwen/Qwen2-VL-2B-Instruct`
- `Qwen/Qwen3-VL-2B-Instruct`

本地离线路径：

- `/tmp/qwen2vl2b_local`
- `/tmp/qwen3vl2b_instruct_local`

围绕这个 backbone，已经真实完成：

- 单卡推理
- 多设备一致性验证
- 本地 toy eval
- one-step LoRA smoke 训练
- LoRA 输出重载
- 配置文件模式训练
- 配置文件模式评测
- 真实 MMEB 子集 `ViDoRe_arxivqa` end-to-end eval

这意味着：

- 第一阶段不是停留在接口层面
- 至少有一个真实 VLM backbone 已经把整个链条真正跑通

对于 `Qwen3-VL-2B-Instruct`，本轮新增完成的真实闭环包括：

- base model toy eval
- one-step LoRA smoke 训练
- LoRA 输出目录重载
- base + adapter 本地 toy eval
- 真实 MMEB 子集 `ViDoRe_arxivqa` base eval

对应产物目录：

- `experiments/stage1_validation/qwen3_vl_full_loop/`

### 第二层：family-loader 真实验证

为了验证我们在 `Nexus/modules/multimodal.py` 里声明支持的 backbone family 不是“只写了分支”，本轮新增了：

- `tools/multimodal_retrieval/validate_backbone_matrix.py`

这个脚本会对每个 family：

1. 检查当前 `transformers` 环境是否存在对应模型类
2. 生成一个 tiny local checkpoint
3. 走真实 `save_pretrained -> load_multimodal_backbone -> from_pretrained` 路径
4. 输出 JSON / Markdown 报告

#### 在当前 `costa` 环境中的结果

环境：

- Python：`/home/szn/zht/miniconda3/envs/costa/bin/python`
- `transformers==4.52.3`

结果：

- `qwen2_vl`：通过
- `qwen2_5_vl`：通过
- `llava_next`：通过
- `qwen3_vl`：不可用

原因：

- `transformers 4.52.3` 中没有 `Qwen3VLForConditionalGeneration`

产物：

- `experiments/stage1_validation/backbone_matrix/costa/report.json`
- `experiments/stage1_validation/backbone_matrix/costa/summary.md`

#### 在额外隔离环境中的结果

为了验证 `qwen3_vl`，本轮额外创建了一个不污染现有环境的临时验证环境：

- `/tmp/nexus_stage1_tf457_env`

其中安装：

- `transformers==4.57.3`

在这个环境中，四个 family 全部通过：

- `qwen2_vl`
- `qwen2_5_vl`
- `qwen3_vl`
- `llava_next`

产物：

- `experiments/stage1_validation/backbone_matrix/transformers_4_57_3/report.json`
- `experiments/stage1_validation/backbone_matrix/transformers_4_57_3/summary.md`

## 二、当前最准确的结论

如果老师问“你们现在适配的是哪个模型”，最准确的说法是：

> 当前完整闭环真实跑通的 backbone 包括 `Qwen2-VL-2B-Instruct` 和 `Qwen3-VL-2B-Instruct`。同时，代码层已面向 `Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Llava-Next` 四个主流多模态 VLM family 做了适配，并且我们已经对这四个 family 做了真实的本地 checkpoint 加载验证；其中 `Qwen3-VL` 的 family 验证和 smoke 闭环运行都在隔离的 `transformers 4.57.3` 环境里完成。

这样说的好处是：

- 真实
- 不夸大
- 也能体现第一阶段已经把后续 backbone 切换的底座准备好了

## 三、老师是否已经明确指定最终 backbone

从老师给出的原始要求，只能严格推出下面这些事实：

- 第二阶段要基于开源 VLM backbone 做训练
- 目标希望超过 `Qwen3-VL-Embedding-8B`

但这段话本身并不能唯一推出：

- 最终必须使用 `Qwen3-VL`
- 或者必须是 `8B`
- 或者必须使用某个固定 checkpoint

所以当前最稳妥的判断是：

- 老师给了一个强基线和方向
- 但没有在任务原文里唯一指定最终 backbone

## 四、这个不确定性会不会影响第一阶段

对第一阶段影响不大。

原因是第一阶段的核心目标是把代码底座整理好，确保支持：

- 多模态微调
- 多模态推理
- 多模态评测
- 数据准备
- backbone 切换时的基本兼容

这些能力不要求一开始就把最终 backbone 唯一确定下来。

只要满足下面两点，第一阶段就算真正完成：

1. 至少一个真实 backbone 完成闭环
2. 其他候选 backbone family 不是写死的，且已经做了真实加载验证

当前这两点都已经满足，而且我们现在已经不止一个 backbone 完成完整 smoke 闭环。

## 五、这个不确定性会影响第二阶段哪些内容

会明显影响下面这些事情：

- 正式训练环境版本
- GPU 显存需求
- batch size 与梯度累积
- 是否只能 LoRA，还是能做更大规模微调
- 图像 / 视频输入长度
- 训练时长与资源预算

尤其要注意：

- `Qwen3-VL` 正式使用时，环境要满足较新的 `transformers`
- 更大参数规模会直接影响显存和训练 recipe

## 六、当前环境策略

### 1. 当前验收环境

本轮第一阶段主体验收使用的是：

- `costa`

该环境里已经确认可用：

- `transformers==4.52.3`
- `accelerate==0.29.1`
- `faiss`
- `pytrec_eval`

它足以支撑：

- `Qwen2-VL` smoke 训练与评测
- `Qwen2.5-VL / Llava-Next` loader family 验证

但它不足以直接支撑：

- `Qwen3-VL` family 验证

### 2. 推荐的正式环境构建方式

仓库内已经准备了：

- `tools/multimodal_retrieval/create_conda_env.sh`
- `tools/multimodal_retrieval/environment.yml`

推荐方式是：

- 新建全新隔离环境
- 安装 `.[eval,multimodal]`
- 安装后直接跑 `validate_stack.sh`

对应脚本：

- `tools/multimodal_retrieval/create_conda_env.sh`

该脚本现在会：

- 自动创建隔离环境
- 安装依赖
- 打印关键版本
- 自动执行验证脚本

对于第二阶段正式训练，更建议直接走这条路径，而不是继续沿用历史遗留的 `costa` 环境。

## 七、最建议问老师的一个问题

如果你现在准备跟老师确认 backbone，我建议直接这样问：

> 老师好，我们第一阶段已经把 Nexus 里的多模态 embedding 训练、推理、评测和 MMEB 数据准备链路整理好了，并且已经用 Qwen2-VL-2B 做了真实闭环验证。接下来为了推进第二阶段正式训练，我想确认一下，您更希望我们优先采用哪条 backbone 路线？是以 Qwen3-VL 系列为主，还是允许我们先用较小模型完成方案验证，再逐步切到更大模型冲榜？

如果想再具体一点，可以继续补一句：

> 另外，您更倾向我们优先做哪个参数规模，例如 2B、4B、8B，还是先不限定，按现有算力条件选择最合适的方案？

## 八、当前可直接汇报的口径

最稳妥的汇报口径如下：

> 第一阶段已经完成。我们已经把 Nexus 整理成一个支持多模态 embedding 微调、推理、评测和 MMEB 数据准备的代码底座。当前完整闭环真实跑通的 backbone 已经包括 Qwen2-VL-2B-Instruct 和 Qwen3-VL-2B-Instruct；同时，Qwen2-VL、Qwen2.5-VL、Qwen3-VL、Llava-Next 四个 backbone family 都已经做了真实加载验证。因此，第一阶段不会因为最终 backbone 尚未唯一确定而被卡住，但第二阶段的正式训练方案仍建议尽快和老师确认。
