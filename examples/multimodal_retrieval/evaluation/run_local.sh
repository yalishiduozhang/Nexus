#!/usr/bin/env bash
set -euo pipefail

python -m Nexus.evaluation.multimodal_retrieval \
  --eval_config examples/multimodal_retrieval/evaluation/eval_config.json \
  --model_config examples/multimodal_retrieval/evaluation/model_config.json

