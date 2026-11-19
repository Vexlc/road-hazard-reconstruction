#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 FU_XI_PROJECT_PATH ASSET_ID" >&2
    exit 2
fi
project="$(realpath "$1")"
name="$2"
if [[ ! "$name" =~ ^(stereo_pothole_[a-zA-Z0-9_]+|photo_manhole_[a-zA-Z0-9_]+)$ ]]; then
    echo "Use a capture asset ID such as stereo_pothole_dataset2_25." >&2
    exit 2
fi
asset="$project/outputs/road_hazards/$name"
samples="$project/work/road_hazard_samples/$name.npz"
if [[ ! -s "$asset/surface.npz" || ! -s "$samples" ]]; then
    echo "Run capture conversion and prepare Gaussian samples first." >&2
    exit 2
fi
python="$project/work/envs/streetcrafter/bin/python"
export PYTHONPATH="$project/work/EasyVolcap"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
cd "$project/work/street_crafter"
export ROAD_HAZARD_ID="$name"
config="$project/outputs/road_hazard_insertions/B_055_$name/render_config.yaml"
"$python" insert_gaussian_assets.py --config configs/pandaset_055_warmup.yaml mode trajectory
"$python" polish_road_hazard.py --config "$config"
"$python" ../road_hazard_pipeline/render_capture_closeups.py --config "$config"
