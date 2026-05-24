#!/usr/bin/env bash
# Convert the exported ONNX model to a TensorRT engine using trtexec.
#
# Requirements:
#   * NVIDIA GPU with a working CUDA + TensorRT installation
#   * `trtexec` available in PATH (ships with the TensorRT installation,
#     or with the official `nvcr.io/nvidia/tensorrt:*-py3` container)
#
# Usage:
#   ./scripts/onnx_to_tensorrt.sh \
#       [onnx_path] [engine_path] [max_seq_len] [opt_batch] [max_batch]
#
# Defaults match this project's ONNX export (max_length=128).

set -euo pipefail

ONNX_PATH="${1:-artifacts/onnx/model.onnx}"
ENGINE_PATH="${2:-artifacts/tensorrt/model.engine}"
SEQ_LEN="${3:-128}"
OPT_BATCH="${4:-8}"
MAX_BATCH="${5:-32}"

if ! command -v trtexec >/dev/null 2>&1; then
    echo "ERROR: trtexec not found in PATH. Install NVIDIA TensorRT or use the official container." >&2
    exit 1
fi

if [[ ! -f "${ONNX_PATH}" ]]; then
    echo "ERROR: ONNX file not found: ${ONNX_PATH}" >&2
    echo "Run 'uv run dtc export-onnx' first." >&2
    exit 1
fi

mkdir -p "$(dirname "${ENGINE_PATH}")"

echo "Converting ${ONNX_PATH} -> ${ENGINE_PATH} (FP16, seq_len=${SEQ_LEN})"

trtexec \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${ENGINE_PATH}" \
    --fp16 \
    --minShapes=input_ids:1x${SEQ_LEN},attention_mask:1x${SEQ_LEN},token_type_ids:1x${SEQ_LEN} \
    --optShapes=input_ids:${OPT_BATCH}x${SEQ_LEN},attention_mask:${OPT_BATCH}x${SEQ_LEN},token_type_ids:${OPT_BATCH}x${SEQ_LEN} \
    --maxShapes=input_ids:${MAX_BATCH}x${SEQ_LEN},attention_mask:${MAX_BATCH}x${SEQ_LEN},token_type_ids:${MAX_BATCH}x${SEQ_LEN} \
    --workspace=4096

echo "TensorRT engine saved to ${ENGINE_PATH}"
