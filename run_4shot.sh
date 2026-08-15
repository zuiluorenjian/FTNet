#!/usr/bin/env bash
set -euo pipefail
source /opt/data/private/anaconda3/etc/profile.d/conda.sh
conda activate FA
cd /opt/data/private/ysb/FTNet

BETA="${1:-15.0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python FTNet.py \
  --config config.yaml \
  --shots 4 \
  --init-beta "$BETA"
