# Multimodal Retrieval Tools

This directory contains helper scripts for preparing MMEB/VLM2Vec-style data for Nexus.

## Scripts

### `convert_vlm2vec_train_to_nexus.py`

Convert training data into Nexus train JSONL.

Example:

```bash
python tools/multimodal_retrieval/convert_vlm2vec_train_to_nexus.py \
  --input /path/to/train.jsonl \
  --source-format llavahound_caption \
  --llavahound-mode caption_retrieval \
  --video-root /path/to/raw/videos \
  --output converted/train.jsonl
```

### `convert_vlm2vec_eval_to_nexus.py`

Convert VLM2Vec pair-style evaluation data into the Nexus local evaluation layout:

- `corpus.jsonl`
- `<split>_queries.jsonl`
- `<split>_qrels.jsonl`

Example:

```bash
python tools/multimodal_retrieval/convert_vlm2vec_eval_to_nexus.py \
  --input /path/to/vlm2vec_eval_export \
  --split test \
  --sequence-mode video \
  --output-dir converted_eval/MSR-VTT
```

### `export_mmeb_v2_inventory.py`

Generate a markdown inventory from the local `VLM2Vec` references.

Example:

```bash
python tools/multimodal_retrieval/export_mmeb_v2_inventory.py \
  --vlm2vec-root ../VLM2Vec \
  --output docs/multimodal_retrieval/MMEB_v2_inventory_generated.md
```

### `export_mmeb_v2_manifest.py`

Generate a machine-readable JSON manifest for the MMEB v2 eval sets and public train sources:

```bash
python tools/multimodal_retrieval/export_mmeb_v2_manifest.py \
  --vlm2vec-root ../VLM2Vec \
  --output docs/multimodal_retrieval/MMEB_v2_manifest.json
```

### `check_idle_gpus.py`

Inspect shared GPUs before starting training:

```bash
python tools/multimodal_retrieval/check_idle_gpus.py
```

### `create_conda_env.sh` and `environment.yml`

Use these to create an isolated runtime instead of installing into the local `base` environment.

### `validate_stack.sh`

Run syntax checks, multimodal tests, inventory export, and conversion smoke tests:

```bash
PYTHON_BIN=/home/szn/zht/miniconda3/envs/costa/bin/python \
bash tools/multimodal_retrieval/validate_stack.sh
```

## Environment

Run these tools inside an isolated environment. Do not install dependencies into the local `base` environment.
