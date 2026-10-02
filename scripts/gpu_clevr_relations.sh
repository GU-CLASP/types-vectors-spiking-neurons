#!/usr/bin/env bash
set -euo pipefail

: "${CLEVR_DIR:?Set CLEVR_DIR to the extracted CLEVR_v1.0 directory}"

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/clevr-resnet50-fpn}"
RELATION_BATCH_SIZE="${RELATION_BATCH_SIZE:-8}"
RELATION_HEAD_BATCH_SIZE="${RELATION_HEAD_BATCH_SIZE:-8192}"
RELATION_EPOCHS="${RELATION_EPOCHS:-10}"
PYTHON="${PYTHON:-python}"

export TORCH_HOME="${TORCH_HOME:-$PWD/artifacts/torch-cache}"
export PYTHONPATH="$PWD/src:$PWD/../pyttr2/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'

for split in train val; do
  "$PYTHON" examples/cache_clevr_relation_features.py \
    --data-dir "$CLEVR_DIR" \
    --split "$split" \
    --object-cache "$OUTPUT_DIR/features/$split" \
    --output "$OUTPUT_DIR/relation-features/$split" \
    --device cuda \
    --batch-size "$RELATION_BATCH_SIZE"
done

"$PYTHON" examples/train_clevr_relations.py \
  --train-cache "$OUTPUT_DIR/relation-features/train" \
  --val-cache "$OUTPUT_DIR/relation-features/val" \
  --output "$OUTPUT_DIR/relations.pt" \
  --device cuda \
  --batch-size "$RELATION_HEAD_BATCH_SIZE" \
  --epochs "$RELATION_EPOCHS"
