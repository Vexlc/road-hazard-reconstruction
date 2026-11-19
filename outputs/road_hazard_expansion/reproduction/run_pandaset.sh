#!/usr/bin/env bash
set -euo pipefail
project="$(realpath "$1")"
work="$(realpath "$2")"
site="$3"
export ROAD_ASSET_IDS="$4"
export ROAD_EXPORT_SCENES="${5:-}"
export PYTHONPATH="$project/work/EasyVolcap:$project/work/asset_pipeline:$project/work/road_hazard_pipeline:$project/work/street_crafter:$work"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 PYTHONUNBUFFERED=1
export ROAD_MESH_CACHE="$work/meshes" ROAD_CAPTURE_CACHE="$work/samples"
cd "$project/work/street_crafter"
../envs/streetcrafter/bin/python "$work/render_pandaset.py" --config "configs/pandaset_${site}_warmup.yaml" mode trajectory
