#!/usr/bin/env bash
set -euo pipefail

python -m Nexus.training.embedder.multimodal_retrieval \
  --model_config examples/multimodal_retrieval/training/model_config.json \
  --data_config examples/multimodal_retrieval/training/data_config.json \
  --training_config examples/multimodal_retrieval/training/training_config.json

