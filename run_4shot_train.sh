#!/usr/bin/env bash
set -euo pipefail
source /opt/data/private/anaconda3/etc/profile.d/conda.sh
conda activate FA
cd /opt/data/private/ysb/FTNet

BETA="${1:-4.0}"
EPOCHS="${2:-20}"
LEARNING_RATE="${3:-0.002}"
BATCH_SIZE="${4:-32}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python FTNet-T.py \
  --config config.yaml \
  --shots 4 \
  --init-beta "$BETA" \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --batch-size "$BATCH_SIZE"
