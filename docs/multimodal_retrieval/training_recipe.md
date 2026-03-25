# Multimodal Embedding Training Recipe

Last updated: 2026-03-24

## Goal

Train a multimodal embedding model in Nexus that is competitive on MMEB v2 and can be iterated quickly across image, video, and visual-document retrieval tasks.

## Backbone Strategy

Recommended priority:

1. Qwen-VL family backbone for the main Nexus path because the codebase now supports Qwen2-VL, Qwen2.5-VL, and Qwen3-VL style loading.
2. Start from a smaller or mid-size checkpoint for fast iteration.
3. Move to a stronger backbone only after the data mixture and evaluation loop are stable.

Candidate directions:

- `Qwen2.5-VL` for early smoke runs and debugging.
- `Qwen3-VL` family for the final competitive path.
- Optional second family only if a clear accuracy or efficiency advantage appears after the first full training loop.

## Training Stages

### Stage A: Image and visdoc warm start

Use:

- `MMEB-train` image subsets
- `vidore/colpali_train_set`
- `openbmb/VisRAG-Ret-Train-In-domain-data`

Purpose:

- stabilize the dual-encoder training path
- verify query and target instruction formats
- validate LoRA target modules and pooling

### Stage B: Add video retrieval and video QA data

Use:

- `video_caption_300k`
- `video_caption_300k-video`
- `video_qa_240k`

Purpose:

- teach the encoder to use temporal visual evidence
- validate raw-video and frame-based input paths

### Stage C: Unified multimodal mixture

Use:

- all Stage A and Stage B sources
- rebalance by modality

Recommended first-pass mixture rules:

- keep image and visdoc batches dominant early to stabilize retrieval quality
- ramp video ratio gradually instead of mixing everything evenly from step 1
- keep the same instruction formatting between train and eval where possible

### Stage D: Hard-negative and distillation pass

Use:

- mined hard negatives from local evaluation runs
- optional teacher scores if a stronger public embedder is used for distillation

Purpose:

- improve fine-grained ranking
- close the gap on retrieval-heavy MMEB subsets

## Model And Loss Settings

Recommended defaults for the first serious run:

- LoRA finetuning first
- `last_token` pooling first, then compare against `mean`
- normalized embeddings enabled
- cross-entropy retrieval loss
- cross-device negatives enabled only after a stable multi-GPU setup is confirmed

Candidate ablations:

- query instruction on vs off
- target instruction on vs off
- `last_token` vs `mean`
- image/visdoc-only warm start vs direct all-modality training
- with and without hard-negative refresh

## Data Preparation Notes

- Keep `media_root`, `image_root`, and `video_root` configurable instead of hardcoding absolute paths into every JSONL.
- Prefer raw `video_path` when the processor supports videos natively.
- Keep a frame-based fallback path for models or environments that do not accept raw video directly.

## Evaluation Plan

### Smoke checks

- local Nexus eval on a tiny converted dataset
- one image task
- one video task
- one visdoc task

### Subset validation

- a small image bundle from MMEB v2
- one video retrieval task and one video QA task
- one visdoc bundle

### Full benchmark

- full MMEB v2 once the environment, data layout, and conversion scripts are stable

## Environment And GPU Safety

- Do not install dependencies into the local `base` environment.
- Use the isolated environment template under `tools/multimodal_retrieval/`.
- Before any run, inspect GPU occupancy and select only idle devices.
- Keep `CUDA_VISIBLE_DEVICES` explicit in training scripts.

## Validated Smoke Command

The local Nexus stack has already passed a one-step offline smoke finetune with a fully local `Qwen/Qwen2-VL-2B-Instruct` backbone copy.

Reference command:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/tmp/hf_nexus_smoke \
HF_DATASETS_CACHE=/tmp/hf_nexus_smoke/datasets \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/home/szn/zht/miniconda3/envs/costa/bin/python -m Nexus.training.embedder.multimodal_retrieval \
  --model_name_or_path /home/szn/zhangx/model_weights/qwen2vl2b_local \
  --processor_name_or_path /home/szn/zhangx/model_weights/qwen2vl2b_local \
  --model_type qwen2_vl \
  --torch_dtype bfloat16 \
  --use_chat_template True \
  --use_lora True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --train_data examples/multimodal_retrieval/data/train.jsonl \
  --media_root examples/multimodal_retrieval/data/media \
  --output_dir /tmp/nexus_mm_smoke_train \
  --overwrite_output_dir \
  --do_train \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --max_steps 1 \
  --train_group_size 2 \
  --query_max_len 256 \
  --passage_max_len 256 \
  --temperature 0.02 \
  --sentence_pooling_method last_token \
  --normalize_embeddings True \
  --remove_unused_columns False \
  --dataloader_drop_last False \
  --report_to none \
  --logging_steps 1 \
  --save_strategy no \
  --bf16
```

Observed smoke result:

- `max_steps=1` completed successfully
- `train_loss = 0.474609375`

## Exit Criteria For Stage 2

- one isolated environment recipe
- one reproducible training command
- one reproducible local evaluation command
- one documented path from raw public data to Nexus training/eval inputs
