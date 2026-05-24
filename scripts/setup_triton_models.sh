#!/usr/bin/env bash
# Copy the exported ONNX model into Triton's model_repository.
#
# Usage:
#   ./scripts/setup_triton_models.sh [onnx_path]
#
# Default onnx_path: artifacts/onnx/model.onnx

set -euo pipefail

ONNX_PATH="${1:-artifacts/onnx/model.onnx}"
MODEL_DIR="triton/model_repository/disaster_tweets_onnx/1"

if [[ ! -f "${ONNX_PATH}" ]]; then
    echo "ERROR: ONNX file not found: ${ONNX_PATH}" >&2
    echo "Run 'uv run dtc export-onnx' first." >&2
    exit 1
fi

mkdir -p "${MODEL_DIR}"
cp "${ONNX_PATH}" "${MODEL_DIR}/model.onnx"

echo "Copied ${ONNX_PATH} -> ${MODEL_DIR}/model.onnx"
echo "Now start Triton:"
echo "  docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\"
echo "      -v \$(pwd)/triton/model_repository:/models \\"
echo "      nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/models"
