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

## Environment

Run these tools inside an isolated environment. Do not install dependencies into the local `base` environment.
