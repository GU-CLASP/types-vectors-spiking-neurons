#!/usr/bin/env bash
set -euo pipefail

: "${CLEVR_DIR:?Set CLEVR_DIR to the extracted CLEVR_v1.0 directory}"

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/clevr-resnet50-fpn}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-8}"
SCENES_PER_SHARD="${SCENES_PER_SHARD:-250}"
HEAD_BATCH_SIZE="${HEAD_BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-10}"
PYTHON="${PYTHON:-python}"

export TORCH_HOME="${TORCH_HOME:-$PWD/artifacts/torch-cache}"
export PYTHONPATH="$PWD/src:$PWD/../pyttr2/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'

for split in train val; do
  "$PYTHON" examples/cache_clevr_features.py \
    --data-dir "$CLEVR_DIR" \
    --split "$split" \
    --output "$OUTPUT_DIR/features/$split" \
    --architecture resnet50-fpn \
    --weights coco \
    --device cuda \
    --batch-size "$FEATURE_BATCH_SIZE" \
    --scenes-per-shard "$SCENES_PER_SHARD"
done

"$PYTHON" examples/train_clevr_heads.py \
  --train-cache "$OUTPUT_DIR/features/train" \
  --val-cache "$OUTPUT_DIR/features/val" \
  --output "$OUTPUT_DIR/heads.pt" \
  --device cuda \
  --batch-size "$HEAD_BATCH_SIZE" \
  --epochs "$EPOCHS"
