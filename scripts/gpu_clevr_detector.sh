#!/usr/bin/env bash
set -euo pipefail

: "${CLEVR_DIR:?Set CLEVR_DIR to the extracted CLEVR_v1.0 directory}"

DETECTOR_DIR="${DETECTOR_DIR:-artifacts/clevr-detector}"
DETECTOR_BATCH_SIZE="${DETECTOR_BATCH_SIZE:-4}"
DETECTOR_WORKERS="${DETECTOR_WORKERS:-4}"
DETECTOR_EPOCHS="${DETECTOR_EPOCHS:-10}"
PYTHON="${PYTHON:-python}"

export TORCH_HOME="${TORCH_HOME:-$PWD/artifacts/torch-cache}"
export PYTHONPATH="$PWD/src:$PWD/../pyttr2/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'

mkdir -p "$DETECTOR_DIR"

resume_args=()
if [[ -f "$DETECTOR_DIR/detector.last.pt" ]]; then
  resume_args=(--resume "$DETECTOR_DIR/detector.last.pt")
fi

"$PYTHON" examples/train_clevr_detector.py \
  --data-dir "$CLEVR_DIR" \
  --output "$DETECTOR_DIR/detector.pt" \
  --device cuda \
  --batch-size "$DETECTOR_BATCH_SIZE" \
  --workers "$DETECTOR_WORKERS" \
  --epochs "$DETECTOR_EPOCHS" \
  "${resume_args[@]}"
