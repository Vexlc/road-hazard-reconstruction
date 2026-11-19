#!/usr/bin/env bash
set -euo pipefail
project="$(realpath "$1")"
export ROAD_BATCH_WORK="$(realpath "$2")"
export ROAD_ONLY_BATCHES="${4:-}"
site="$3"
export PYTHONPATH="$project/work/EasyVolcap:$project/work/asset_pipeline:$project/work/street_crafter"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$project/work/street_crafter"
"${STREETCRAFTER_PYTHON:-../envs/streetcrafter/bin/python}" "$ROAD_BATCH_WORK/render_batches.py" --config "configs/pandaset_${site}_warmup.yaml" mode trajectory
