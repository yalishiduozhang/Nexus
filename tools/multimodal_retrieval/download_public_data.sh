#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$PWD/data}"
RAW_ROOT="${RAW_ROOT:-$DATA_ROOT/raw}"
DOWNLOAD_TRAIN="${DOWNLOAD_TRAIN:-1}"
DOWNLOAD_EVAL="${DOWNLOAD_EVAL:-1}"
DRY_RUN="${DRY_RUN:-1}"

TRAIN_ROOT="${RAW_ROOT}/vlm2vec_train"
EVAL_ROOT="${RAW_ROOT}/vlm2vec_eval"

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

clone_hf_git_lfs_repo() {
  local repo_url="$1"
  local target_dir="$2"

  if [[ -d "${target_dir}/.git" ]]; then
    echo "Repo already exists: ${target_dir}"
    return 0
  fi

  run_cmd git clone "${repo_url}" "${target_dir}"
  if [[ -d "${target_dir}/.git" ]]; then
    (
      cd "${target_dir}"
      run_cmd git lfs pull
    )
  fi
}

echo "DATA_ROOT=${DATA_ROOT}"
echo "RAW_ROOT=${RAW_ROOT}"
echo "DRY_RUN=${DRY_RUN}"

mkdir -p "${TRAIN_ROOT}" "${EVAL_ROOT}"

if [[ "${DOWNLOAD_TRAIN}" == "1" ]]; then
  echo "== Training sources =="
  clone_hf_git_lfs_repo "https://huggingface.co/datasets/TIGER-Lab/MMEB-train" "${TRAIN_ROOT}/MMEB-train"
  clone_hf_git_lfs_repo "https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction" "${TRAIN_ROOT}/ShareGPTVideo"
  clone_hf_git_lfs_repo "https://huggingface.co/datasets/vidore/colpali_train_set" "${TRAIN_ROOT}/colpali_train_set"
  clone_hf_git_lfs_repo "https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data" "${TRAIN_ROOT}/VisRAG-Ret-Train-In-domain-data"

  echo "Post-processing reminders:"
  echo "- MMEB-train may require running unzip helpers from the dataset repo."
  echo "- ShareGPTVideo archives may need unpacking under train_300k / train_600k."
  echo "- If needed, derive video_qa_240k.jsonl from video_240k_caption_15k.jsonl."
fi

if [[ "${DOWNLOAD_EVAL}" == "1" ]]; then
  echo "== Evaluation source =="
  clone_hf_git_lfs_repo "https://huggingface.co/datasets/TIGER-Lab/MMEB-V2" "${EVAL_ROOT}/MMEB-V2"

  echo "Post-processing reminders:"
  echo "- unpack image, video, and visdoc archives before Nexus conversion"
  echo "- preserve the raw folder layout for traceability"
fi

echo "Finished. Review storage usage before switching DRY_RUN=0."
