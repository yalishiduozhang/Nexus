#!/usr/bin/env bash
set -euo pipefail

if [[ "${REQUIRE_EXPLICIT_GPUS:-1}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to an idle GPU before training."
  echo "Use tools/multimodal_retrieval/check_idle_gpus.py first on shared machines."
  exit 1
fi

python -m Nexus.training.embedder.multimodal_retrieval \
  --model_config examples/multimodal_retrieval/training/model_config.json \
  --data_config examples/multimodal_retrieval/training/data_config.json \
  --training_config examples/multimodal_retrieval/training/training_config.json
