#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TMP_ROOT="${TMP_ROOT:-/tmp/nexus_mmeb_validation}"

rm -rf "${TMP_ROOT}"
mkdir -p "${TMP_ROOT}"

echo "[1/4] py_compile"
"${PYTHON_BIN}" -m py_compile \
  "${REPO_ROOT}/Nexus/__init__.py" \
  "${REPO_ROOT}/Nexus/abc/__init__.py" \
  "${REPO_ROOT}/Nexus/abc/inference/__init__.py" \
  "${REPO_ROOT}/Nexus/abc/inference/inference_engine.py" \
  "${REPO_ROOT}/Nexus/evaluation/text_retrieval/evaluator.py" \
  "${REPO_ROOT}/Nexus/evaluation/multimodal_retrieval/__main__.py" \
  "${REPO_ROOT}/Nexus/evaluation/multimodal_retrieval/__init__.py" \
  "${REPO_ROOT}/Nexus/evaluation/multimodal_retrieval/arguments.py" \
  "${REPO_ROOT}/Nexus/evaluation/multimodal_retrieval/data_loader.py" \
  "${REPO_ROOT}/Nexus/evaluation/multimodal_retrieval/runner.py" \
  "${REPO_ROOT}/Nexus/evaluation/text_retrieval/data_loader.py" \
  "${REPO_ROOT}/Nexus/inference/__init__.py" \
  "${REPO_ROOT}/Nexus/inference/embedder/__init__.py" \
  "${REPO_ROOT}/Nexus/inference/embedder/multimodal_retrieval/generic.py" \
  "${REPO_ROOT}/Nexus/modules/multimodal.py" \
  "${REPO_ROOT}/Nexus/training/embedder/multimodal_retrieval/__main__.py" \
  "${REPO_ROOT}/Nexus/training/embedder/multimodal_retrieval/dataset.py" \
  "${REPO_ROOT}/tools/multimodal_retrieval/convert_vlm2vec_train_to_nexus.py" \
  "${REPO_ROOT}/tools/multimodal_retrieval/convert_vlm2vec_eval_to_nexus.py" \
  "${REPO_ROOT}/tools/multimodal_retrieval/export_mmeb_v2_inventory.py" \
  "${REPO_ROOT}/tools/multimodal_retrieval/export_mmeb_v2_manifest.py" \
  "${REPO_ROOT}/tools/multimodal_retrieval/hf_dataset_manager.py" \
  "${REPO_ROOT}/tools/multimodal_retrieval/prepare_mmeb_v2_eval_data.py" \
  "${REPO_ROOT}/tools/multimodal_retrieval/prepare_mmeb_v2_train_data.py" \
  "${REPO_ROOT}/tools/multimodal_retrieval/prepare_public_data.py" \
  "${REPO_ROOT}/tools/multimodal_retrieval/vlm2vec_manifest_lib.py" \
  "${REPO_ROOT}/tools/multimodal_retrieval/check_idle_gpus.py"

echo "[2/4] pytest"
cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m pytest tests/multimodal_retrieval -q

echo "[3/4] inventory export"
"${PYTHON_BIN}" tools/multimodal_retrieval/export_mmeb_v2_inventory.py \
  --vlm2vec-root /home/szn/zhangx/explore/VLM2Vec \
  --output "${TMP_ROOT}/mmeb_inventory_generated.md"
"${PYTHON_BIN}" tools/multimodal_retrieval/export_mmeb_v2_manifest.py \
  --vlm2vec-root /home/szn/zhangx/explore/VLM2Vec \
  --output "${TMP_ROOT}/mmeb_manifest_generated.json"

echo "[4/4] conversion smoke"
cat > "${TMP_ROOT}/train_pair.jsonl" <<'EOF'
{"query_text":"find clip","query_image":{"paths":["frames/q1.jpg","frames/q2.jpg"],"bytes":[null,null]},"pos_text":"positive","pos_image":{"paths":["frames/p1.jpg","frames/p2.jpg"],"bytes":[null,null]},"neg_text":[],"neg_image":[],"global_dataset_name":"video/msrvtt"}
EOF

cat > "${TMP_ROOT}/eval_pair.jsonl" <<'EOF'
{"query_text":["query one"],"query_image":[null],"cand_text":[["target text"]],"cand_image":[[{"paths":["doc/page1.png"],"bytes":[null]}]],"dataset_infos":{"cand_names":["d1"],"label_name":"d1"}}
EOF

"${PYTHON_BIN}" tools/multimodal_retrieval/convert_vlm2vec_train_to_nexus.py \
  --input "${TMP_ROOT}/train_pair.jsonl" \
  --source-format vlm2vec_pairs \
  --sequence-mode video \
  --dataset-name MSR-VTT \
  --output "${TMP_ROOT}/converted_train.jsonl"

"${PYTHON_BIN}" tools/multimodal_retrieval/convert_vlm2vec_eval_to_nexus.py \
  --input "${TMP_ROOT}/eval_pair.jsonl" \
  --sequence-mode image \
  --dataset-name toy_eval \
  --output-dir "${TMP_ROOT}/converted_eval"

"${PYTHON_BIN}" tools/multimodal_retrieval/prepare_mmeb_v2_eval_data.py \
  --manifest "${REPO_ROOT}/docs/multimodal_retrieval/MMEB_v2_manifest.json" \
  --raw-root "${TMP_ROOT}/raw_eval" \
  --output-root "${TMP_ROOT}/prepared_eval" \
  --datasets HatefulMemes MSVD \
  --write-eval-configs-dir "${TMP_ROOT}/prepared_eval_configs" \
  --allow-missing \
  --dry-run

echo "Validation completed successfully."
