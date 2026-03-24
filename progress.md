# Progress Log

## 2026-03-24

### Completed before this round

- Audited `Nexus`, `FlagEmbedding`, and `VLM2Vec` to decide the integration direction.
- Implemented the first multimodal retrieval stack in Nexus.
- Added example train/infer/eval configs and scripts.
- Created the first local milestone commit:
  - `8a5d364` `Add multimodal embedding pipeline and examples`
- Produced shareable archive artifacts for reporting.

### Review findings recorded in this round

- Raw `video` / `video_path` / `videos` inputs were not supported end to end.
- `Qwen3-VL` model type was not registered in the generic multimodal loading path.
- Local evaluation assumed media files always lived under the JSONL directory.
- The multimodal embedder accepted multiple devices but only used the first one.

### Work in progress in this round

- Added project-level planning and logging documents:
  - `plan.md`
  - `progress.md`
- Hardened the multimodal utility layer for MMEB v2:
  - native video normalization
  - raw video fallback decoding
  - chat-template updates for video items
  - Qwen3-VL registration
- Restored actual multi-device inference behavior in the multimodal embedder.
- Updated training and evaluation path handling so converted datasets can keep separate media roots.
- Added data preparation assets:
  - `docs/multimodal_retrieval/MMEB_v2_data_inventory.md`
  - `tools/multimodal_retrieval/convert_vlm2vec_train_to_nexus.py`
  - `tools/multimodal_retrieval/convert_vlm2vec_eval_to_nexus.py`
  - `tools/multimodal_retrieval/export_mmeb_v2_inventory.py`
  - `tools/multimodal_retrieval/README.md`
- Added execution-prep assets:
  - `docs/multimodal_retrieval/training_recipe.md`
  - `tools/multimodal_retrieval/check_idle_gpus.py`
  - `tools/multimodal_retrieval/create_conda_env.sh`
  - `tools/multimodal_retrieval/environment.yml`
- Hardened example training scripts so shared-GPU runs require explicit `CUDA_VISIBLE_DEVICES`.
- Updated example docs and eval config to reflect the new media-root and video-input support.
- Ran syntax checks on all touched Python files with `python -m py_compile`.

### Pending after this round

- Commit the readiness pass as the next local milestone.
- Build isolated runtime environments for actual training or evaluation jobs.
- Execute the conversion scripts and training/evaluation workflows in an environment that includes `transformers`, `datasets`, and `PyYAML`.
