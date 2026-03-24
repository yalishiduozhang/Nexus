#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-nexus-mmeb}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

echo "Creating isolated conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -e ".[eval,multimodal]"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  echo "Installing flash-attn inside ${ENV_NAME}"
  conda run -n "${ENV_NAME}" python -m pip install flash-attn --no-build-isolation
fi

echo "Environment ready: ${ENV_NAME}"
echo "Do not install multimodal dependencies into the local base environment."
