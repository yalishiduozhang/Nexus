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
  - `docs/multimodal_retrieval/MMEB_v2_manifest.json`
  - `tools/multimodal_retrieval/convert_vlm2vec_train_to_nexus.py`
  - `tools/multimodal_retrieval/convert_vlm2vec_eval_to_nexus.py`
  - `tools/multimodal_retrieval/export_mmeb_v2_inventory.py`
  - `tools/multimodal_retrieval/export_mmeb_v2_manifest.py`
  - `tools/multimodal_retrieval/README.md`
- Added execution-prep assets:
  - `docs/multimodal_retrieval/training_recipe.md`
  - `docs/multimodal_retrieval/data_collection_playbook.md`
  - `tools/multimodal_retrieval/check_idle_gpus.py`
  - `tools/multimodal_retrieval/create_conda_env.sh`
  - `tools/multimodal_retrieval/download_public_data.sh`
  - `tools/multimodal_retrieval/environment.yml`
- Hardened example training scripts so shared-GPU runs require explicit `CUDA_VISIBLE_DEVICES`.
- Updated example docs and eval config to reflect the new media-root and video-input support.
- Ran syntax checks on all touched Python files with `python -m py_compile`.
- Fixed package import coupling so multimodal modules can be imported without unrelated `onnx` dependencies being installed:
  - lazy imports in `Nexus/__init__.py`
  - lazy imports in `Nexus/abc/__init__.py`
  - lazy imports in `Nexus/abc/inference/__init__.py`
  - lazy imports in `Nexus/inference/__init__.py`
  - lazy imports in `Nexus/inference/embedder/__init__.py`
  - lazy imports in `Nexus/evaluation/multimodal_retrieval/__init__.py`
- Downgraded `onnx` / `onnxruntime` / `tensorrt` in the generic inference engine to optional imports instead of hard import requirements.
- Added regression tests under `tests/multimodal_retrieval/` covering:
  - multimodal normalization with video fields
  - prefixed query/pos parsing for video inputs
  - evaluation media-root resolution
  - conversion-tool helper behavior
- Fixed cache-path behavior so multimodal and text retrieval evaluation loaders prefer writable cache locations and fall back to `/tmp`.
- Fixed conversion tools so they also use writable datasets cache directories and can auto-fallback to the only available split when local JSON/JSONL loads as `train`.
- Added `tools/multimodal_retrieval/validate_stack.sh` to reproduce the validation flow end to end.
- Verified with the isolated `costa` environment:
  - `import Nexus` and multimodal module import now succeed
  - `export_mmeb_v2_inventory.py` runs successfully
  - `convert_vlm2vec_train_to_nexus.py` CLI runs successfully
  - `convert_vlm2vec_eval_to_nexus.py` CLI runs successfully
  - `pytest tests/multimodal_retrieval -q` passes
  - `validate_stack.sh` passes end to end

### Pending after this round

- Build isolated runtime environments for actual training or evaluation jobs.
- Execute the conversion scripts and training/evaluation workflows in an environment that includes `transformers`, `datasets`, and `PyYAML`.
- Decide whether to start large-scale public data download locally or leave actual download to the data-collection owner with the provided manifests and scripts.
- Run real training and benchmark jobs on idle GPUs only.

## 2026-03-25

### Additional codebase work completed

- Added manifest/export helper library:
  - `tools/multimodal_retrieval/vlm2vec_manifest_lib.py`
- Added HF HTTP dataset planner/downloader:
  - `tools/multimodal_retrieval/hf_dataset_manager.py`
- Added manifest-driven public-data orchestrator:
  - `tools/multimodal_retrieval/prepare_public_data.py`
- Added local batch train-data converter with stage-config emission:
  - `tools/multimodal_retrieval/prepare_mmeb_v2_train_data.py`
- Added config-file CLI parsing for multimodal train/eval entrypoints:
  - `Nexus/training/embedder/multimodal_retrieval/__main__.py`
  - `Nexus/evaluation/multimodal_retrieval/__main__.py`
- Added a bundled example train/eval dataset under `examples/multimodal_retrieval/data/`.
- Further reduced eager optional imports on the text-retrieval side so multimodal config parsing does not require unrelated backends up front.

### Tooling fixes completed

- Fixed the train/eval conversion CLIs so local input directories can be scanned recursively for parquet/json/jsonl shards instead of only the top level.
- Fixed `prepare_public_data.py` so it can consume an older manifest file and auto-augment missing download metadata.
- Fixed `hf_dataset_manager.py` dataset URL construction for Hugging Face tree/resolve endpoints.
- Added `--input` support to `check_idle_gpus.py` so GPU selection can still be computed from a pre-captured `nvidia-smi` CSV dump when direct probing fails inside an isolated environment.

### Validation completed in this round

- `pytest tests/multimodal_retrieval -q` passes with 17 tests.
- `tools/multimodal_retrieval/validate_stack.sh` passes end to end in the isolated `costa` environment.
- Regenerated:
  - `docs/multimodal_retrieval/MMEB_v2_manifest.json`
  - `docs/multimodal_retrieval/MMEB_v2_inventory_generated.md`

### Real public-data smoke result

- Verified that the public-data path is executable on a real MMEB train subset.
- Planned, downloaded, and converted:
  - source: `TIGER-Lab/MMEB-train / HatefulMemes / original`
  - raw artifact: `/tmp/nexus_public_smoke/raw/vlm2vec_train/MMEB-train/HatefulMemes/original-00000-of-00001.parquet`
  - Nexus train output: `/tmp/nexus_public_smoke/nexus/train/image/HatefulMemes.jsonl`
- Verified that `prepare_mmeb_v2_train_data.py` can transform the downloaded raw subset into stage-ready outputs and generated stage configs:
  - `/tmp/nexus_public_smoke/stage_ready/image/HatefulMemes.jsonl`
  - `/tmp/nexus_public_smoke/stage_configs/`

### Current blockers after this round

- Python-level Hugging Face access inside the sandbox still fails for SDK-style networking, so actual public-data download in this environment requires unsandboxed execution.
- Full public data collection still exceeds the free disk currently available on this machine, so selective download remains the only safe local path right now.
- No real GPU finetuning run has started yet because CUDA is not available inside the current isolated sandbox runtime, and shared-GPU jobs still need to be launched only on idle devices in an unsandboxed runtime.
