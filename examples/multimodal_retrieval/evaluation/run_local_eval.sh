#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./outputs/mm_embedder_qwen25vl_lora}"
DATASET_DIR="${DATASET_DIR:-./data/eval}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-./outputs/mm_eval}"

python -m Nexus.evaluation.multimodal_retrieval \
  --eval_name "${EVAL_NAME:-mmeb_local}" \
  --dataset_dir "${DATASET_DIR}" \
  --eval_output_dir "${EVAL_OUTPUT_DIR}" \
  --eval_output_path "${EVAL_OUTPUT_DIR}/summary.md" \
  --embedder_name_or_path "${MODEL_NAME_OR_PATH}" \
  --processor_name_or_path "${MODEL_NAME_OR_PATH}" \
  --embedder_batch_size "${EMBED_BATCH_SIZE:-8}" \
  --embedder_query_max_length "${QUERY_MAX_LEN:-512}" \
  --embedder_passage_max_length "${PASSAGE_MAX_LEN:-1024}" \
  --pooling_method "${POOLING_METHOD:-last_token}" \
  --normalize_embeddings True \
  --search_top_k "${SEARCH_TOP_K:-1000}"
